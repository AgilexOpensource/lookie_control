import os

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

from kstone_mock.ik import IkConfig, WholeBodyIK, quat_error  # noqa: E402
from kstone_mock.model import build_robot_model  # noqa: E402
from kstone_mock.transforms import make_transform, split_transform  # noqa: E402

MODEL = "lookie__pf-lookie_trio_r01__ee-revo2std_r01"
DESCRIPTION = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "lookie_description"))
TORSO = [f"torso_joint{i}" for i in range(1, 5)]
ARMS = [f"{s}_arm_joint{i}" for s in "lr" for i in range(1, 8)]


@pytest.fixture(scope="module")
def robot():
    urdf = os.path.join(DESCRIPTION, "urdf", MODEL + ".urdf")
    if not os.path.exists(urdf):
        pytest.skip("lookie_description not found next to lookie_control")
    with open(os.path.join(DESCRIPTION, "config", MODEL + "_initial.yaml"), encoding="utf-8") as f:
        home = {k: float(v) for k, v in yaml.safe_load(f)["zeros"].items()}
    built = build_robot_model(urdf, {"lookie_description": DESCRIPTION}, {"left_arm": (1, 0, 0, 1)})
    return built, home


def test_model_structure(robot):
    built, _ = robot
    names = [built.model.joint(i).name for i in range(built.model.njnt)]
    assert names[:3] == ["base_x", "base_y", "base_yaw"]
    assert built.root_body == "chassis_base_link"
    assert {m.joint for m in built.mimics} >= {"left_index_distal_joint", "right_thumb_distal_joint"}
    assert built.marker_bodies == {"left_arm": "marker_left_arm"}
    assert built.model.body("marker_left_arm").mocapid[0] >= 0


LEFT = [f"l_arm_link{i}" for i in range(3, 8)] + ["l_flange", "l_flange1", "left_base_link"]
RIGHT = [f"r_arm_link{i}" for i in range(3, 8)] + ["r_flange", "r_flange1", "right_base_link"]
COLLISION_PAIRS = [(LEFT, RIGHT), (LEFT[1:], ["torso_link3", "torso_link4"]),
                   (RIGHT[1:], ["torso_link3", "torso_link4"])]


def _setup(robot, **config):
    built, home = robot
    model = built.model
    data = mujoco.MjData(model)
    for name, value in home.items():
        try:
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        except KeyError:
            pass
    joints = TORSO + ARMS
    ik = WholeBodyIK(
        model, joints, {"left": "left_base_link", "right": "right_base_link"}, home,
        posture_weights={j: 1e-3 for j in joints},
        velocity_weights={j: (0.02 if j in TORSO else 1e-3) for j in joints},
        config=IkConfig(**{"joint_velocity_limits": {j: 1.0 for j in TORSO},
                           "joint_acceleration_limits": {j: 3.0 for j in TORSO}, **config}),
    )
    mujoco.mj_kinematics(model, data)
    return model, data, ik


@pytest.mark.parametrize("pairs", [[], COLLISION_PAIRS], ids=["no_collision", "collision"])
def test_ik_converges_with_torso(robot, pairs):
    model, data, ik = _setup(robot, collision_pairs=pairs)
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    torso0 = data.qpos[[model.jnt_qposadr[model.joint(j).id] for j in TORSO]].copy()

    # Reach forward and down with both hands: needs the torso to help
    offset = make_transform([0.25, 0.0, -0.15], [1.0, 0.0, 0.0, 0.0])
    targets = {}
    for task, (p, q) in start.items():
        t = make_transform(p, q)
        t[:3, 3] += offset[:3, 3]
        targets[task] = split_transform(t)
    for _ in range(600):
        ik.solve(data, targets, 0.01)
    for task, (pos_err, rot_err) in ik.errors(data, targets).items():
        assert pos_err < 0.01, task
        assert rot_err < 0.05, task
    torso = data.qpos[[model.jnt_qposadr[model.joint(j).id] for j in TORSO]]
    assert np.max(np.abs(torso - torso0)) > 1e-3


def test_limits_and_frozen_joints(robot):
    model, data, ik = _setup(robot)
    head = model.jnt_qposadr[model.joint("head_joint1").id]
    data.qpos[head] = 0.3
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    # Unreachable target: joints must stop at the margin, never past the bound
    targets = {t: (p + np.array([2.0, 0.0, 1.0]), q) for t, (p, q) in start.items()}
    previous = data.qpos.copy()
    for _ in range(300):
        ik.solve(data, targets, 0.01)
        step = np.abs(data.qpos - previous)[ik.qpos_index]
        assert np.all(step <= ik.max_velocity * 0.01 + 1e-6)
        previous = data.qpos.copy()
    for jid in (model.joint(j).id for j in TORSO + ARMS):
        lo, hi = model.jnt_range[jid]
        q = data.qpos[model.jnt_qposadr[jid]]
        assert lo - 1e-6 <= q <= hi + 1e-6
    assert data.qpos[head] == pytest.approx(0.3)


def test_self_collision_keeps_hands_apart(robot):
    def hand_gap(**config):
        model, data, ik = _setup(robot, **config)
        (pl, ql), (pr, qr) = ik.body_pose(data, "left"), ik.body_pose(data, "right")
        middle = (pl + pr) / 2 + np.array([0.15, 0.0, 0.0])
        for _ in range(300):
            ik.solve(data, {"left": (middle, ql), "right": (middle, qr)}, 0.01)
        return np.linalg.norm(ik.body_pose(data, "left")[0] - ik.body_pose(data, "right")[0])

    assert hand_gap() < 0.02
    assert hand_gap(collision_pairs=COLLISION_PAIRS) > 0.04


def _torso_run(model, data, ik, target_fn, steps):
    index = [model.jnt_qposadr[model.joint(j).id] for j in TORSO]
    q, errors = [], []
    for k in range(steps):
        targets = target_fn(k * 0.01)
        ik.solve(data, targets, 0.01)
        q.append(data.qpos[index].copy())
        errors.append(max(e[0] for e in ik.errors(data, targets).values()))
    v = np.diff(np.array(q), axis=0) / 0.01
    return np.abs(v).max(), np.abs(np.diff(v, axis=0) / 0.01).max(), np.array(errors)


def test_hand_jerk_barely_moves_torso(robot):
    model, data, ik = _setup(robot)
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    jump = np.array([0.15, 0.0, 0.0])

    def targets(t):
        return {k: (p + (jump if t < 0.1 else 0.0), q) for k, (p, q) in start.items()}

    v, a, _ = _torso_run(model, data, ik, targets, 100)
    assert v < 0.5
    assert a <= 3.0 * 1.01  # QP solver tolerance


@pytest.mark.parametrize("torso_task", [False, True])
def test_turn_ramps_torso_and_tracks(robot, torso_task):
    # The default torso pose weights (height/pitch locked, yaw 0.01) must not stop turning.
    extra = dict(torso_body="torso_link4", torso_root="chassis_base_link",
                 torso_position_cost=(0.0, 0.0, 3.0),
                 torso_orientation_cost=(3.0, 3.0, 0.01)) if torso_task else {}
    model, data, ik = _setup(robot, **extra)
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    jid = model.joint("torso_joint4").id
    centre, axis = data.xanchor[jid].copy(), data.xaxis[jid].copy()

    def targets(t):
        quat, rot = np.zeros(4), np.zeros(9)
        mujoco.mju_axisAngle2Quat(quat, axis, min(t, np.pi / 4))
        mujoco.mju_quat2Mat(rot, quat)
        out = {}
        for k, (p, q) in start.items():
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, quat, q)
            out[k] = (centre + rot.reshape(3, 3) @ (p - centre), turned)
        return out

    v, a, errors = _torso_run(model, data, ik, targets, 250)
    assert data.qpos[model.jnt_qposadr[jid]] > 0.3
    assert a <= 3.0 * 1.01  # QP solver tolerance
    assert errors.max() < 0.01


def test_reset_clears_acceleration_state(robot):
    _, data, ik = _setup(robot)
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    for _ in range(20):
        ik.solve(data, {k: (p + np.array([0.0, 0.3, 0.0]), q) for k, (p, q) in start.items()}, 0.01)
    assert np.any(ik._acceleration_limit.velocity != 0.0)
    ik.reset()
    assert np.all(ik._acceleration_limit.velocity == 0.0)


@pytest.mark.parametrize("weight", [0.0, 3.0])
def test_torso_pose_task(robot, weight):
    model, data, ik = _setup(robot, torso_body="torso_link4",
                             torso_root="chassis_base_link",
                             torso_position_cost=(0.0, 0.0, weight),
                             torso_orientation_cost=(weight, weight, 0.0))
    torso = model.body("torso_link4").id
    q0 = data.xquat[torso].copy()
    z0 = data.xpos[torso][2]
    start = {t: ik.body_pose(data, t) for t in ("left", "right")}
    targets = {k: (p + np.array([0.2, 0.0, -0.15]), q) for k, (p, q) in start.items()}
    for _ in range(300):
        ik.solve(data, targets, 0.01)
    assert max(e[0] for e in ik.errors(data, targets).values()) < 0.01
    tilt = np.linalg.norm(quat_error(data.xquat[torso], q0))
    dz = abs(data.xpos[torso][2] - z0)
    if weight:
        assert tilt < np.radians(2.0) and dz < 0.01
    else:
        assert tilt > np.radians(10.0) and dz > 0.05


FOLLOW = dict(torso_body="torso_link4", torso_root="chassis_base_link",
              torso_position_cost=(1.0, 0.0, 3.0), torso_orientation_cost=(3.0, 3.0, 0.3),
              torso_gain=0.3)


@pytest.mark.parametrize("delta,axis,angle", [
    ([0.0, 0.0, -0.15], [0, 1, 0], 0.0),   # squat
    ([0.08, 0.0, 0.0], [0, 1, 0], 0.0),    # forward
    ([0.0, 0.0, 0.0], [0, 1, 0], 0.35),    # lean (pitch)
    ([0.0, 0.0, 0.0], [0, 0, 1], 0.7),     # turn (yaw)
], ids=["squat", "forward", "pitch", "yaw"])
def test_torso_target_tracks(robot, delta, axis, angle):
    # Headset-follow use case: a rate-limited torso target, hands carried along rigidly.
    model, data, ik = _setup(robot, **FOLLOW)
    p0, q0 = ik.torso_pose(data)
    np.testing.assert_allclose(p0, ik.torso_home.translation(), atol=1e-6)
    np.testing.assert_allclose(q0, ik.torso_home.rotation().wxyz, atol=1e-6)
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, np.asarray(axis, float), angle)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, dq)
    centre = data.xpos[model.body("torso_link4").id].copy()
    hands = {}
    for k in ("left", "right"):
        p, q = ik.body_pose(data, k)
        turned = np.zeros(4)
        mujoco.mju_mulQuat(turned, dq, q)
        hands[k] = (centre + np.asarray(delta) + rot.reshape(3, 3) @ (p - centre), turned)
    ramp = 50
    for step in range(400):
        s = min(1.0, (step + 1) / ramp)
        dq_s = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq_s, np.asarray(axis, float), angle * s)
        target_q = np.zeros(4)
        mujoco.mju_mulQuat(target_q, dq_s, q0)
        ik.set_torso_target(p0 + s * np.asarray(delta), target_q)
        ik.solve(data, hands, 0.01)
    p, q = ik.torso_pose(data)
    assert np.linalg.norm(p - (p0 + np.asarray(delta))) < 0.01
    assert np.linalg.norm(quat_error(q, target_q)) < np.radians(2.0)
    assert max(e[0] for e in ik.errors(data, hands).values()) < 0.01
