"""Stand-in for kstone_ros: MuJoCo whole-body IK with a built-in viewer.

The node exposes the task-space contract that lookie_teleop talks to and that
the real kstone_ros is expected to implement later (see README.md).
"""

import math
import os
import threading

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import yaml

from .ik import IkConfig, WholeBodyIK
from .model import BASE_JOINTS, build_robot_model
from .transforms import make_transform, split_transform

MARKER_COLOURS = ((1.0, 0.3, 0.2, 0.9), (0.2, 0.5, 1.0, 0.9), (0.2, 0.9, 0.3, 0.9))


def _pose_to_arrays(msg) -> tuple[np.ndarray, np.ndarray]:
    p, q = msg.position, msg.orientation
    quat = np.array([q.w, q.x, q.y, q.z], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("invalid quaternion")
    position = np.array([p.x, p.y, p.z], dtype=float)
    if not np.all(np.isfinite(position)):
        raise ValueError("invalid position")
    return position, quat / norm


def _fill_pose(msg, position: np.ndarray, quat: np.ndarray) -> None:
    msg.position.x, msg.position.y, msg.position.z = (float(v) for v in position)
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (float(v) for v in quat)


class KstoneMockNode(Node):
    def __init__(self) -> None:
        super().__init__("kstone_mock")
        p = self.declare_parameter
        p("description_package", "lookie_description")
        p("description_dir", "")
        p("model_name", "lookie__pf-lookie_trio_r01__ee-revo2std_r01")
        p("urdf_path", "")
        p("initial_pose_path", "")
        p("prefix", "/kstone")
        p("base_frame", "chassis_base_link")
        p("world_frame", "odom")
        p("control_rate_hz", 100.0)
        p("publish_rate_hz", 50.0)
        p("task_ids", ["left_arm", "right_arm"])
        p("task_bodies", ["left_base_link", "right_base_link"])
        p("ik_joints", [f"torso_joint{i}" for i in range(1, 5)]
          + [f"l_arm_joint{i}" for i in range(1, 8)] + [f"r_arm_joint{i}" for i in range(1, 8)])
        p("slow_joints", [f"torso_joint{i}" for i in range(1, 5)])
        p("slow_joint_velocity_weight", 0.02)
        p("velocity_weight", 1e-3)
        p("posture_weight", 1e-3)
        p("posture_gain", 0.5)
        p("position_weight", 1.0)
        p("orientation_weight", 0.4)
        p("damping", 1e-4)
        p("max_joint_velocity", 3.0)
        p("slow_joint_max_velocity", 1.0)
        p("slow_joint_max_acceleration", 3.0)
        p("torso_body", "torso_link4")
        p("torso_task_id", "torso")
        p("torso_position_weight", [1.0, 0.0, 3.0])
        p("torso_orientation_weight", [3.0, 3.0, 0.3])
        p("torso_gain", 0.3)
        p("ik_iterations", 2)
        p("limit_margin", 0.01)
        p("limit_gain", 0.95)
        p("ik_solver", "daqp")
        # Each entry "body_a1 body_a2 ... | body_b1 ...": keep the two groups apart
        p("collision_pairs", [""])
        p("collision_min_distance", 0.01)
        p("collision_detection_distance", 0.1)
        p("extra_joint_max_velocity", 4.0)
        p("max_linear_velocity", 0.8)
        p("max_angular_velocity", 1.2)
        p("cmd_vel_timeout_s", 0.5)
        p("viewer", True)
        p("forward_to_driver", False)
        p("driver_command_topic", "/trio/control/joint_states")
        self._v = lambda name: self.get_parameter(name).value

        description_dir = self._v("description_dir") or self._share(self._v("description_package"))
        model_name = self._v("model_name")
        urdf_path = self._v("urdf_path") or os.path.join(description_dir, "urdf", model_name + ".urdf")
        initial_path = self._v("initial_pose_path") or os.path.join(
            description_dir, "config", model_name + "_initial.yaml")
        with open(initial_path, encoding="utf-8") as stream:
            self._home = {k: float(v) for k, v in (yaml.safe_load(stream) or {}).get("zeros", {}).items()}

        self._tasks = list(self._v("task_ids"))
        bodies = list(self._v("task_bodies"))
        if len(self._tasks) != len(bodies):
            raise ValueError("task_ids and task_bodies must have equal length")
        markers = {t: MARKER_COLOURS[i % len(MARKER_COLOURS)] for i, t in enumerate(self._tasks)}
        self.get_logger().info(f"building MuJoCo model from {urdf_path}")
        self._robot = build_robot_model(
            urdf_path, {self._v("description_package"): description_dir}, markers)
        self._model = self._robot.model
        self._data = mujoco.MjData(self._model)

        ik_joints = list(self._v("ik_joints"))
        slow = set(self._v("slow_joints"))
        self._ik = WholeBodyIK(
            self._model, ik_joints, dict(zip(self._tasks, bodies)), self._home,
            posture_weights={j: float(self._v("posture_weight")) for j in ik_joints},
            velocity_weights={j: float(self._v("slow_joint_velocity_weight") if j in slow
                                       else self._v("velocity_weight")) for j in ik_joints},
            config=IkConfig(
                position_weight=float(self._v("position_weight")),
                orientation_weight=float(self._v("orientation_weight")),
                damping=float(self._v("damping")),
                posture_gain=float(self._v("posture_gain")),
                max_joint_velocity=float(self._v("max_joint_velocity")),
                joint_velocity_limits={j: float(self._v("slow_joint_max_velocity")) for j in slow},
                joint_acceleration_limits=(
                    {j: float(self._v("slow_joint_max_acceleration")) for j in slow}
                    if float(self._v("slow_joint_max_acceleration")) > 0.0 else {}),
                iterations=int(self._v("ik_iterations")),
                limit_margin=float(self._v("limit_margin")),
                limit_gain=float(self._v("limit_gain")),
                solver=str(self._v("ik_solver")),
                collision_pairs=self._collision_pairs(self._v("collision_pairs")),
                collision_min_distance=float(self._v("collision_min_distance")),
                collision_detection_distance=float(self._v("collision_detection_distance")),
                torso_body=str(self._v("torso_body")),
                torso_root=str(self._v("base_frame")),
                torso_position_cost=tuple(float(c) for c in self._v("torso_position_weight")),
                torso_orientation_cost=tuple(float(c) for c in self._v("torso_orientation_weight")),
                torso_gain=float(self._v("torso_gain")),
            ),
        )

        mimic_joints = {m.joint for m in self._robot.mimics}
        self._published_joints = [j for j in self._robot.urdf_joints if j not in mimic_joints]
        self._extra_joints = [
            j for j in self._published_joints if j not in ik_joints and "wheel" not in j]
        self._qadr = {
            self._model.joint(i).name: int(self._model.jnt_qposadr[i]) for i in range(self._model.njnt)}
        self._extra_target: dict[str, float] = {}
        self._mocap_ids = {
            t: int(self._model.body(b).mocapid[0]) for t, b in self._robot.marker_bodies.items()}
        self._lock = threading.Lock()
        self._cmd_vel = np.zeros(3)
        self._cmd_vel_stamp = -math.inf

        self._reset_joints()
        self._home_targets = {t: self._task_pose_base(t) for t in self._tasks}
        self._targets = dict(self._home_targets)
        # The torso task lives in the IK's torso_root frame, which is base_frame.
        self._torso_id = str(self._v("torso_task_id")) if self._ik.torso_task is not None else ""
        if self._torso_id:
            home = self._ik.torso_home
            self._torso_home = (np.array(home.translation()), np.array(home.rotation().wxyz))
            self._torso_target = self._torso_home

        prefix = self._v("prefix").rstrip("/")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pose_pubs, self._home_pubs = {}, {}
        for task in self._tasks:
            base = f"{prefix}/tasks/{task}"
            self._pose_pubs[task] = self.create_publisher(PoseStamped, base + "/pose", 10)
            self._home_pubs[task] = self.create_publisher(PoseStamped, base + "/home_pose", latched)
            self.create_subscription(
                PoseStamped, base + "/target", lambda msg, t=task: self._on_target(t, msg), 10)
            self.create_service(
                Trigger, base + "/reset_target",
                lambda req, res, t=task: self._on_reset_target(t, res))
        if self._torso_id:
            base = f"{prefix}/tasks/{self._torso_id}"
            self._torso_pose_pub = self.create_publisher(PoseStamped, base + "/pose", 10)
            self._torso_home_pub = self.create_publisher(PoseStamped, base + "/home_pose", latched)
            self.create_subscription(PoseStamped, base + "/target", self._on_torso_target, 10)
            self.create_service(Trigger, base + "/reset_target", self._on_reset_torso)
        self._joint_pub = self.create_publisher(JointState, prefix + "/joint_states", 10)
        self._base_pub = self.create_publisher(PoseStamped, prefix + "/base_pose", 10)
        self._driver_pub = (self.create_publisher(JointState, self._v("driver_command_topic"), 10)
                            if self._v("forward_to_driver") else None)
        self.create_subscription(JointState, prefix + "/extra_joint_commands", self._on_extra, 10)
        self.create_subscription(Twist, prefix + "/cmd_vel", self._on_cmd_vel, 10)
        self.create_service(Trigger, prefix + "/reset_home", self._on_reset_home)
        self._publish_home()

        self._viewer = None
        if self._v("viewer"):
            try:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(
                    self._model, self._data, show_left_ui=False, show_right_ui=False)
            except Exception as exc:  # noqa: BLE001 - viewer is optional
                self.get_logger().warning(f"MuJoCo viewer unavailable, running headless: {exc}")

        self._dt = 1.0 / float(self._v("control_rate_hz"))
        self._last_tick = self._now()
        self.create_timer(self._dt, self._tick)
        self.create_timer(1.0 / float(self._v("publish_rate_hz")), self._publish)
        self.get_logger().info(
            f"kstone_mock ready: tasks={self._tasks}, prefix={prefix}, "
            f"forward_to_driver={bool(self._driver_pub)}")

    @staticmethod
    def _collision_pairs(entries: list[str]) -> list[tuple[list[str], list[str]]]:
        pairs = []
        for entry in entries:
            if not entry.strip():
                continue
            groups = entry.split("|")
            if len(groups) != 2:
                raise ValueError(f"collision_pairs entry needs exactly one '|': {entry!r}")
            pairs.append((groups[0].split(), groups[1].split()))
        return pairs

    @staticmethod
    def _share(package: str) -> str:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(package)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _reset_joints(self) -> None:
        self._data.qpos[:] = 0.0
        for name, value in self._home.items():
            if name in self._qadr:
                self._data.qpos[self._qadr[name]] = value
        self._extra_target = {j: self._home.get(j, 0.0) for j in self._extra_joints}
        self._apply_mimics()
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
        self._ik.reset()

    def _apply_mimics(self) -> None:
        q = self._data.qpos
        for mimic in self._robot.mimics:
            q[self._qadr[mimic.joint]] = mimic.multiplier * q[self._qadr[mimic.source]] + mimic.offset

    def _base_transform(self) -> np.ndarray:
        q = self._data.qpos
        x, y, yaw = (q[self._qadr[j]] for j in BASE_JOINTS)
        t = np.eye(4)
        t[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        t[:2, 3] = x, y
        return t

    def _task_pose_base(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        world = make_transform(*self._ik.body_pose(self._data, task))
        return split_transform(np.linalg.inv(self._base_transform()) @ world)

    # Callbacks --------------------------------------------------------------
    def _on_target(self, task: str, msg: PoseStamped) -> None:
        try:
            position, quat = _pose_to_arrays(msg.pose)
        except ValueError as exc:
            self.get_logger().warning(f"rejected {task} target: {exc}", throttle_duration_sec=1.0)
            return
        frame = msg.header.frame_id or self._v("base_frame")
        with self._lock:
            if frame == self._v("world_frame"):
                position, quat = split_transform(
                    np.linalg.inv(self._base_transform()) @ make_transform(position, quat))
            elif frame != self._v("base_frame"):
                self.get_logger().warning(
                    f"rejected {task} target in unsupported frame '{frame}'", throttle_duration_sec=1.0)
                return
            self._targets[task] = (position, quat)

    def _on_torso_target(self, msg: PoseStamped) -> None:
        try:
            position, quat = _pose_to_arrays(msg.pose)
        except ValueError as exc:
            self.get_logger().warning(f"rejected torso target: {exc}", throttle_duration_sec=1.0)
            return
        frame = msg.header.frame_id or self._v("base_frame")
        if frame != self._v("base_frame"):
            self.get_logger().warning(
                f"rejected torso target in unsupported frame '{frame}'", throttle_duration_sec=1.0)
            return
        with self._lock:
            self._torso_target = (position, quat)

    def _on_reset_torso(self, request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._torso_target = self._torso_home
        response.success, response.message = True, "torso target reset to home"
        return response

    def _on_extra(self, msg: JointState) -> None:
        with self._lock:
            for name, value in zip(msg.name, msg.position):
                if name in self._extra_target and math.isfinite(value):
                    self._extra_target[name] = float(value)

    def _on_cmd_vel(self, msg: Twist) -> None:
        values = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        if not np.all(np.isfinite(values)):
            return
        lin, ang = float(self._v("max_linear_velocity")), float(self._v("max_angular_velocity"))
        with self._lock:
            self._cmd_vel = np.clip(values, [-lin, -lin, -ang], [lin, lin, ang])
            self._cmd_vel_stamp = self._now()

    def _on_reset_target(self, task: str, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._targets[task] = self._home_targets[task]
        response.success, response.message = True, f"{task} target reset to home"
        return response

    def _on_reset_home(self, request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            base = {j: self._data.qpos[self._qadr[j]] for j in BASE_JOINTS}
            self._reset_joints()
            for name, value in base.items():
                self._data.qpos[self._qadr[name]] = value
            self._targets = dict(self._home_targets)
            if self._torso_id:
                self._torso_target = self._torso_home
            self._cmd_vel[:] = 0.0
        response.success, response.message = True, "joints and targets reset to home"
        return response

    # Loops ------------------------------------------------------------------
    def _tick(self) -> None:
        now = self._now()
        dt = min(max(now - self._last_tick, 0.0), 0.1)
        self._last_tick = now
        with self._lock:
            q = self._data.qpos
            if now - self._cmd_vel_stamp > float(self._v("cmd_vel_timeout_s")):
                self._cmd_vel[:] = 0.0
            vx, vy, wz = self._cmd_vel
            yaw = q[self._qadr["base_yaw"]]
            q[self._qadr["base_x"]] += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
            q[self._qadr["base_y"]] += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
            q[self._qadr["base_yaw"]] = math.remainder(yaw + wz * dt, math.tau)

            step = float(self._v("extra_joint_max_velocity")) * dt
            for name, target in self._extra_target.items():
                jid = self._model.joint(name).id
                if self._model.jnt_limited[jid]:
                    target = float(np.clip(target, *self._model.jnt_range[jid]))
                adr = self._qadr[name]
                q[adr] += float(np.clip(target - q[adr], -step, step))
            self._apply_mimics()

            mujoco.mj_kinematics(self._model, self._data)
            base = self._base_transform()
            world_targets = {
                t: split_transform(base @ make_transform(*pose)) for t, pose in self._targets.items()}
            if self._torso_id:
                self._ik.set_torso_target(*self._torso_target)
            self._ik.solve(self._data, world_targets, dt)
            for task, mocap in self._mocap_ids.items():
                position, quat = world_targets[task]
                self._data.mocap_pos[mocap], self._data.mocap_quat[mocap] = position, quat

            if self._viewer is not None:
                if self._viewer.is_running():
                    with self._viewer.lock():
                        mujoco.mj_forward(self._model, self._data)
                    self._viewer.sync()
                else:
                    self._viewer = None

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            poses = {t: self._task_pose_base(t) for t in self._tasks}
            torso = self._ik.torso_pose(self._data) if self._torso_id else None
            positions = [float(self._data.qpos[self._qadr[j]]) for j in self._published_joints]
            base = split_transform(self._base_transform())
        for task, (position, quat) in poses.items():
            msg = PoseStamped()
            msg.header.stamp, msg.header.frame_id = stamp, self._v("base_frame")
            _fill_pose(msg.pose, position, quat)
            self._pose_pubs[task].publish(msg)
        if torso is not None:
            msg = PoseStamped()
            msg.header.stamp, msg.header.frame_id = stamp, self._v("base_frame")
            _fill_pose(msg.pose, *torso)
            self._torso_pose_pub.publish(msg)
        joints = JointState()
        joints.header.stamp = stamp
        joints.name, joints.position = list(self._published_joints), positions
        self._joint_pub.publish(joints)
        if self._driver_pub is not None:
            driver = JointState()
            driver.header.stamp = stamp
            pairs = [(n, v) for n, v in zip(joints.name, joints.position) if "wheel" not in n]
            driver.name, driver.position = [n for n, _ in pairs], [v for _, v in pairs]
            self._driver_pub.publish(driver)
        base_msg = PoseStamped()
        base_msg.header.stamp, base_msg.header.frame_id = stamp, self._v("world_frame")
        _fill_pose(base_msg.pose, *base)
        self._base_pub.publish(base_msg)

    def _publish_home(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for task, (position, quat) in self._home_targets.items():
            msg = PoseStamped()
            msg.header.stamp, msg.header.frame_id = stamp, self._v("base_frame")
            _fill_pose(msg.pose, position, quat)
            self._home_pubs[task].publish(msg)
        if self._torso_id:
            msg = PoseStamped()
            msg.header.stamp, msg.header.frame_id = stamp, self._v("base_frame")
            _fill_pose(msg.pose, *self._torso_home)
            self._torso_home_pub.publish(msg)

    def destroy_node(self) -> bool:
        if self._viewer is not None:
            self._viewer.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KstoneMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
