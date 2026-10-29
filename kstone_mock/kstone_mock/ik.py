"""Whole-body differential IK on a MuJoCo model, solved as a QP with mink.

Each sub-step solves for a joint velocity ``v`` (``dq = v * dt``) over the
selected joints (torso + both arms):

    min  sum_t |W_t (J_t dq - e_t)|^2 + sum_j v_j dq_j^2
         + sum_j p_j (dq_j - k (q_nom_j - q_j))^2 + lambda |dq|^2
    s.t. joint position limits, per-joint velocity limits,
         per-joint acceleration limits (|v_k - v_{k-1}| <= a_max dt),
         optional self-collision distance limits,
         dq_j = 0 for every joint that is not an IK joint

``v_j`` penalises joint motion only (no steady-state bias); a larger value on
the torso makes the arms move first, so the torso mainly contributes when an
arm alone cannot reach the target. ``p_j`` is a weak pull toward the nominal
posture that resolves redundancy.

The acceleration limit is what keeps fast hand motion off the torso: a hand
jerk saturates the arm joints for a few ticks, but the torso can only ramp its
velocity slowly, so short spikes barely move it. A sustained demand such as
turning lets it reach its full velocity within ``v_max / a_max`` seconds.

Unlike a clipped least-squares step, limits
are hard QP constraints, so the solver redistributes motion to other joints
instead of stalling at a bound.
"""

import copy
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np


@dataclass
class IkConfig:
    position_weight: float = 1.0
    orientation_weight: float = 0.4
    damping: float = 1e-4
    posture_gain: float = 0.5
    default_posture_weight: float = 1e-3
    default_velocity_weight: float = 1e-3
    max_joint_velocity: float = 3.0
    # Per-joint overrides (rad/s, rad/s^2); joints missing from the acceleration
    # map have no acceleration limit.
    joint_velocity_limits: dict[str, float] = field(default_factory=dict)
    joint_acceleration_limits: dict[str, float] = field(default_factory=dict)
    iterations: int = 2
    limit_margin: float = 0.01
    limit_gain: float = 0.95
    solver: str = "daqp"
    # Each entry is (bodies_a, bodies_b): keep every geom of ``bodies_a`` away
    # from every geom of ``bodies_b``. Empty disables collision avoidance.
    collision_pairs: list[tuple[list[str], list[str]]] = field(default_factory=list)
    collision_min_distance: float = 0.01
    collision_detection_distance: float = 0.1
    # Pose task on a torso body relative to ``torso_root`` (the base), target =
    # its pose at the nominal posture. Costs are per axis of the body frame
    # (position x, y, z; orientation roll, pitch, yaw); 0 leaves that axis
    # free. An empty body or all-zero costs disables it.
    torso_body: str = ""
    torso_root: str = ""
    torso_position_cost: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torso_orientation_cost: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Fraction of the torso error corrected per solve. Below 1 the torso
    # converges over several ticks instead of demanding error/dt velocity,
    # which overshoots and oscillates against the acceleration limit.
    torso_gain: float = 1.0


def quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``target`` (wxyz)."""
    inverse = np.zeros(4)
    mujoco.mju_negQuat(inverse, current)
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, target, inverse)
    if delta[0] < 0.0:
        delta = -delta
    error = np.zeros(3)
    mujoco.mju_quat2Vel(error, delta, 1.0)
    return error


class AccelerationLimit(mink.Limit):
    """Bound the change of joint velocity between consecutive solves."""

    def __init__(self, nv: int, dofs: np.ndarray, max_acceleration: np.ndarray) -> None:
        self.dofs = np.asarray(dofs, dtype=int)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        self.velocity = np.zeros(len(self.dofs))
        projection = np.eye(nv)[self.dofs]
        self._G = np.vstack([projection, -projection])

    def reset(self) -> None:
        self.velocity[:] = 0.0

    def compute_qp_inequalities(self, configuration: mink.Configuration, dt: float) -> mink.Constraint:
        del configuration
        upper = (self.velocity + self.max_acceleration * dt) * dt
        lower = (self.velocity - self.max_acceleration * dt) * dt
        return mink.Constraint(G=self._G, h=np.concatenate([upper, -lower]))


def _body_geoms(model: mujoco.MjModel, bodies: list[str]) -> list[int]:
    ids = {model.body(name).id for name in bodies}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids]


class WholeBodyIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        joints: list[str],
        task_bodies: dict[str, str],
        nominal: dict[str, float],
        posture_weights: dict[str, float] | None = None,
        velocity_weights: dict[str, float] | None = None,
        config: IkConfig | None = None,
    ) -> None:
        self.model = model
        self.config = cfg = config or IkConfig()
        self.joints = list(joints)
        joint_ids = [model.joint(name).id for name in self.joints]
        for name, jid in zip(self.joints, joint_ids):
            if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"IK joint {name} must be hinge or slide")
        self.qpos_index = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
        self.dof_index = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
        self.task_bodies = {task: model.body(body).id for task, body in task_bodies.items()}

        # mink owns its MjData; collision geoms need contype/conaffinity set, so
        # in that case it gets a private model copy (contacts stay disabled).
        ik_model = model
        if cfg.collision_pairs:
            ik_model = copy.deepcopy(model)
            ik_model.geom_contype[:] = 0
            ik_model.geom_conaffinity[:] = 0
        geom_pairs = []
        for bodies_a, bodies_b in cfg.collision_pairs:
            geoms_a, geoms_b = _body_geoms(ik_model, bodies_a), _body_geoms(ik_model, bodies_b)
            ik_model.geom_contype[geoms_a + geoms_b] = 1
            ik_model.geom_conaffinity[geoms_a + geoms_b] = 1
            geom_pairs.append((geoms_a, geoms_b))
        self._configuration = mink.Configuration(ik_model)

        self._frame_tasks = {
            task: mink.FrameTask(
                body, "body",
                position_cost=cfg.position_weight, orientation_cost=cfg.orientation_weight,
            )
            for task, body in task_bodies.items()
        }
        posture_weights = posture_weights or {}
        velocity_weights = velocity_weights or {}
        posture_cost = np.zeros(model.nv)
        velocity_cost = np.zeros(model.nv)
        for name, dof in zip(self.joints, self.dof_index):
            posture_cost[dof] = np.sqrt(float(posture_weights.get(name, cfg.default_posture_weight)))
            velocity_cost[dof] = np.sqrt(float(velocity_weights.get(name, cfg.default_velocity_weight)))
        self._posture_task = mink.PostureTask(ik_model, cost=posture_cost, gain=cfg.posture_gain)
        target = self._configuration.q.copy()
        target[self.qpos_index] = [float(nominal.get(name, 0.0)) for name in self.joints]
        self._posture_task.set_target(target)
        self._damping_task = mink.DampingTask(ik_model, cost=velocity_cost)

        frozen = sorted(set(range(model.nv)) - set(self.dof_index.tolist()))
        self._constraints = [mink.DofFreezingTask(ik_model, dof_indices=frozen)] if frozen else []

        # Only bound the IK joints: frozen joints (head, hands, mimics) are set
        # by the node and may sit on or past a bound, which would make the QP
        # infeasible.
        position_limit = mink.ConfigurationLimit(
            ik_model, gain=cfg.limit_gain, min_distance_from_limits=cfg.limit_margin)
        ik_dofs = np.intersect1d(position_limit.indices, self.dof_index)
        position_limit.indices = ik_dofs
        position_limit.projection_matrix = np.eye(model.nv)[ik_dofs] if len(ik_dofs) else None
        self.max_velocity = np.array([
            float(cfg.joint_velocity_limits.get(name, cfg.max_joint_velocity)) for name in self.joints])
        self._limits = [
            position_limit,
            mink.VelocityLimit(ik_model, dict(zip(self.joints, self.max_velocity))),
        ]
        accel = [(dof, float(cfg.joint_acceleration_limits[name]))
                 for name, dof in zip(self.joints, self.dof_index)
                 if name in cfg.joint_acceleration_limits]
        self._acceleration_limit = None
        if accel:
            dofs, values = zip(*accel)
            self._acceleration_limit = AccelerationLimit(model.nv, np.array(dofs), np.array(values))
        if geom_pairs:
            self._limits.append(mink.CollisionAvoidanceLimit(
                ik_model, geom_pairs,
                minimum_distance_from_collisions=cfg.collision_min_distance,
                collision_detection_distance=cfg.collision_detection_distance,
            ))
        self._tasks = [self._posture_task, self._damping_task]

        self.torso_task = None
        costs = tuple(cfg.torso_position_cost) + tuple(cfg.torso_orientation_cost)
        if cfg.torso_body and cfg.torso_root and any(c > 0.0 for c in costs):
            self.torso_task = mink.RelativeFrameTask(
                cfg.torso_body, "body", cfg.torso_root, "body",
                position_cost=np.asarray(cfg.torso_position_cost, dtype=float),
                orientation_cost=np.asarray(cfg.torso_orientation_cost, dtype=float),
                gain=float(cfg.torso_gain))
            nominal_config = mink.Configuration(ik_model, target)
            self.torso_home = nominal_config.get_transform(
                cfg.torso_body, "body", cfg.torso_root, "body")
            self.torso_task.set_target(self.torso_home)
            self._tasks.append(self.torso_task)

    def torso_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Current torso pose (m, wxyz) in the ``torso_root`` frame."""
        body = self.model.body(self.config.torso_body).id
        root = self.model.body(self.config.torso_root).id
        rot_root = data.xmat[root].reshape(3, 3)
        position = rot_root.T @ (data.xpos[body] - data.xpos[root])
        quat, inv_root = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv_root, data.xquat[root])
        mujoco.mju_mulQuat(quat, inv_root, data.xquat[body])
        if quat[0] < 0.0:
            quat = -quat
        return position, quat

    def set_torso_target(self, position: np.ndarray, quat: np.ndarray) -> None:
        """Torso pose target (m, wxyz) in the ``torso_root`` frame."""
        if self.torso_task is None:
            raise RuntimeError("torso task is disabled")
        self.torso_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(quat, dtype=float)), np.asarray(position, dtype=float)))

    def reset(self) -> None:
        """Forget the previous joint velocity, e.g. after teleporting ``qpos``."""
        if self._acceleration_limit is not None:
            self._acceleration_limit.reset()

    def body_pose(self, data: mujoco.MjData, task: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.task_bodies[task]
        return data.xpos[body].copy(), data.xquat[body].copy()

    def errors(self, data: mujoco.MjData, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[float, float]]:
        """Position (m) and orientation (rad) error per task."""
        result = {}
        for task, (position, quat) in targets.items():
            current_p, current_q = self.body_pose(data, task)
            result[task] = (
                float(np.linalg.norm(position - current_p)),
                float(np.linalg.norm(quat_error(quat, current_q))),
            )
        return result

    def solve(
        self,
        data: mujoco.MjData,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        dt: float,
    ) -> None:
        """Move ``data.qpos`` toward the world-frame task targets for one tick."""
        cfg = self.config
        steps = max(1, cfg.iterations)
        sub_dt = max(dt, 1e-4) / steps
        tasks = list(self._tasks)
        for task, (position, quat) in targets.items():
            frame_task = self._frame_tasks[task]
            frame_task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3(np.asarray(quat, dtype=float)), np.asarray(position, dtype=float)))
            tasks.append(frame_task)

        configuration = self._configuration
        configuration.update(data.qpos)
        accel = self._acceleration_limit
        for _ in range(steps):
            velocity = self._solve(tasks, sub_dt, accel)
            if velocity is None and accel is not None:
                # Near a position limit the acceleration bound can conflict
                # with the braking the position limit asks for.
                velocity = self._solve(tasks, sub_dt, None)
            if velocity is None:
                if accel is not None:
                    accel.reset()
                break
            if accel is not None:
                accel.velocity[:] = velocity[accel.dofs]
            configuration.integrate_inplace(velocity, sub_dt)
        data.qpos[self.qpos_index] = configuration.q[self.qpos_index]
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)

    def _solve(self, tasks, dt: float, accel: AccelerationLimit | None) -> np.ndarray | None:
        limits = self._limits + ([accel] if accel is not None else [])
        try:
            return mink.solve_ik(
                self._configuration, tasks, dt, self.config.solver, damping=self.config.damping,
                safety_break=False, limits=limits, constraints=self._constraints,
            )
        except mink.NoSolutionFound:
            return None
