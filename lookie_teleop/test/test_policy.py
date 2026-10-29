import math

import numpy as np
import pytest

from lookie_teleop.policy import (
    ArmClutch, ArmMappingConfig, ButtonPress, ChassisConfig, ClutchGate, InputFrame, Joy,
    TeleopConfig, TeleopPolicy, apply_deadzone, map_chassis)
from lookie_teleop.transforms import matrix_to_quat, quat_to_matrix, rot_z, yaw_of


def pose(p=(0.0, 0.0, 0.0), r=None):
    t = np.eye(4)
    t[:3, 3] = p
    if r is not None:
        t[:3, :3] = r
    return t


def joy(grip=0.0, trigger=0.0, stick=(0.0, 0.0), primary=0, secondary=0):
    return Joy([stick[0], stick[1], trigger, grip], [0, int(grip > 0.5), 0, primary, secondary])


def test_deadzone():
    assert apply_deadzone(0.1, 0.15) == 0.0
    assert apply_deadzone(1.0, 0.15) == pytest.approx(1.0)
    assert apply_deadzone(-0.575, 0.15) == pytest.approx(-0.5)


def test_button_short_and_long():
    b = ButtonPress(1.0)
    assert b.update(True, 0.0) is None
    assert b.update(False, 0.2) == "short"
    assert b.update(True, 1.0) is None
    assert b.update(True, 2.1) == "long"
    assert b.update(True, 3.0) is None
    assert b.update(False, 3.1) is None


def test_clutch_hysteresis_and_block():
    c = ClutchGate("hold", 0.6, 0.3)
    assert not c.update(0.5, 0.0)
    assert c.update(0.7, 0.1)
    assert c.update(0.4, 0.2)
    assert not c.update(0.2, 0.3)
    c.update(0.9, 0.4)
    c.block_until_release()
    assert not c.update(0.9, 0.5)
    assert not c.update(0.1, 0.6)
    assert c.update(0.9, 0.7)


def test_clutch_double_press():
    c = ClutchGate("double_press", 0.6, 0.3)
    assert not c.update(1.0, 0.0)
    assert not c.update(0.0, 0.1)
    assert c.update(1.0, 0.2)
    assert c.update(1.0, 2.0)
    # Single long hold does not engage
    c = ClutchGate("double_press", 0.6, 0.3)
    c.update(1.0, 0.0)
    assert not c.update(1.0, 1.0)
    assert not c.update(0.0, 1.1)


def test_arm_clutch_heading_and_rotation():
    cfg = ArmMappingConfig(smoothing=1.0, workspace_min=(-5, -5, -5), workspace_max=(5, 5, 5))
    arm = ArmClutch(cfg)
    ee0 = pose((0.3, 0.2, 0.6), quat_to_matrix([0.5, -0.5, 0.5, -0.5]))
    # Operator faces +Y in the VR world
    yaw = math.pi / 2
    c0 = pose((1.0, 1.0, 1.0))
    arm.engage(c0, ee0, yaw)
    np.testing.assert_allclose(arm.update(c0), ee0, atol=1e-9)

    # Moving along the operator's forward (+Y world) moves the EE along robot +X
    target = arm.update(pose((1.0, 1.1, 1.0)))
    np.testing.assert_allclose(target[:3, 3], [0.4, 0.2, 0.6], atol=1e-9)

    # Rolling the controller about the operator's forward rolls the EE about robot +X
    roll = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # 90 deg about world Y
    target = arm.update(pose((1.0, 1.0, 1.0), roll))
    expected_rel = rot_z(yaw).T @ roll @ rot_z(yaw)
    np.testing.assert_allclose(target[:3, :3], expected_rel @ ee0[:3, :3], atol=1e-9)
    np.testing.assert_allclose(expected_rel, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-9)


def test_arm_clutch_limits():
    arm = ArmClutch(ArmMappingConfig(smoothing=1.0, max_offset_m=0.1, jump_reject_m=0.0,
                                     workspace_min=(0, 0, 0), workspace_max=(1, 1, 0.65)))
    arm.engage(pose(), pose((0.3, 0.2, 0.6)), 0.0)
    target = arm.update(pose((0.0, 0.0, 1.0)))
    np.testing.assert_allclose(target[:3, 3], [0.3, 0.2, 0.65], atol=1e-9)
    target = arm.update(pose((1.0, 0.0, 0.0)))
    np.testing.assert_allclose(target[:3, 3], [0.4, 0.2, 0.6], atol=1e-9)


def test_smoothing_converges():
    arm = ArmClutch(ArmMappingConfig(smoothing=0.5))
    arm.engage(pose(), pose((0.3, 0.2, 0.6)), 0.0)
    for _ in range(40):
        target = arm.update(pose((0.1, 0.0, 0.0)))
    np.testing.assert_allclose(target[:3, 3], [0.4, 0.2, 0.6], atol=1e-6)


def test_arm_speed_caps():
    arm = ArmClutch(ArmMappingConfig(smoothing=1.0, max_linear_speed=1.0, max_angular_speed=2.0,
                                     jump_reject_m=0.0))
    arm.engage(pose(), pose((0.3, 0.2, 0.6)), 0.0)
    arm.update(pose(), 0.0)
    target = arm.update(pose((0.5, 0.0, 0.0), rot_z(1.0)), 0.01)
    np.testing.assert_allclose(target[:3, 3], [0.31, 0.2, 0.6], atol=1e-9)
    assert math.atan2(target[1, 0], target[0, 0]) == pytest.approx(0.02, abs=1e-6)
    for k in range(2, 200):
        target = arm.update(pose((0.5, 0.0, 0.0), rot_z(1.0)), 0.01 * k)
    np.testing.assert_allclose(target[:3, 3], [0.8, 0.2, 0.6], atol=1e-9)


def test_arm_jump_rejection_reanchors():
    arm = ArmClutch(ArmMappingConfig(smoothing=1.0, jump_reject_m=0.15))
    arm.engage(pose(), pose((0.3, 0.2, 0.6)), 0.0)
    arm.update(pose((0.05, 0.0, 0.0)))
    # Tracking glitch: the pose teleports 0.5 m, the target stays put...
    target = arm.update(pose((0.55, 0.0, 0.0)))
    np.testing.assert_allclose(target[:3, 3], [0.35, 0.2, 0.6], atol=1e-9)
    # ...and later motion continues relative to the new anchor
    target = arm.update(pose((0.6, 0.0, 0.0)))
    np.testing.assert_allclose(target[:3, 3], [0.4, 0.2, 0.6], atol=1e-9)


def test_yaw_of():
    assert yaw_of(rot_z(0.7)) == pytest.approx(0.7)
    assert yaw_of(rot_z(0.7), (0.0, 1.0, 0.0)) == pytest.approx(0.7 + math.pi / 2)


def test_chassis_mapping():
    cfg = ChassisConfig(deadzone=0.0, max_linear=0.5, max_angular=1.0)
    axes = {"left": [0.0, -1.0, 0, 0], "right": [1.0, 0.0, 0, 0]}
    vx, vy, wz = map_chassis(axes, cfg)
    assert vx == pytest.approx(0.5)   # stick forward -> drive forward
    assert vy == pytest.approx(0.0)
    assert wz == pytest.approx(-1.0)  # stick right -> turn clockwise
    assert map_chassis({"left": None, "right": None}, cfg) == (0.0, 0.0, 0.0)


def frame(now, grip=0.0, controller=None, ee=None, **kw):
    return InputFrame(
        now=now,
        controllers={"left": controller, "right": None},
        joys={"left": joy(grip=grip, **kw), "right": joy()},
        ee_poses={"left": ee, "right": None},
    )


def test_policy_engage_home_and_stale():
    policy = TeleopPolicy(TeleopConfig(start_enabled=True, view_mode="first_person",
                                       arm=ArmMappingConfig(smoothing=1.0)))
    ee = pose((0.3, 0.2, 0.6))
    out = policy.step(frame(0.0, grip=0.0, controller=pose(), ee=ee))
    assert out.arm_targets == {}
    out = policy.step(frame(0.1, grip=1.0, controller=pose(), ee=ee))
    np.testing.assert_allclose(out.arm_targets["left"], ee)
    out = policy.step(frame(0.2, grip=1.0, controller=pose((0.0, 0.0, 0.1)), ee=ee))
    assert out.arm_targets["left"][2, 3] == pytest.approx(0.7)

    # Stale controller disengages; when fresh again it re-latches the current EE
    out = policy.step(frame(0.3, grip=1.0, controller=None, ee=ee))
    assert not policy.arms["left"].engaged and out.arm_targets == {}
    out = policy.step(frame(0.4, grip=1.0, controller=pose((0.0, 0.0, 0.5)), ee=ee))
    np.testing.assert_allclose(out.arm_targets["left"], ee)



def test_policy_enable_toggle_and_disabled_outputs():
    policy = TeleopPolicy(TeleopConfig(start_enabled=False))
    ee = pose((0.3, 0.2, 0.6))
    f = frame(0.0, grip=1.0, controller=pose(), ee=ee, stick=(0.0, -1.0))
    out = policy.step(f)
    assert out.arm_targets == {} and out.cmd_vel == (0.0, 0.0, 0.0) and out.hand_positions == {}

    def right(now, a=0, b=0):
        return InputFrame(now=now, joys={"left": None, "right": joy(primary=a, secondary=b)})

    # A before calibration is refused
    policy.step(right(0.5, a=1))
    out = policy.step(right(0.6))
    assert out.events == ["calibration_required"] and not policy.enabled
    # B long press calibrates, then A starts
    policy.step(right(1.0, b=1))
    out = policy.step(right(2.1, b=1))
    assert out.events == ["calibrate"] and policy.calibrated and not policy.enabled
    policy.step(right(2.15))
    policy.step(right(2.2, a=1))
    out = policy.step(right(2.3))
    assert out.events == ["enabled"] and policy.enabled
    # A while running does nothing
    policy.step(right(2.4, a=1))
    assert policy.step(right(2.5)).events == [] and policy.enabled
    out = policy.step(frame(2.6, grip=0.0, controller=pose(), ee=ee, stick=(0.0, -1.0), trigger=1.0))
    assert out.cmd_vel[0] > 0.0
    assert out.hand_positions["left"][2] == pytest.approx(1.41)


def test_hand_tracking_latches_without_reference():
    policy = TeleopPolicy(TeleopConfig(start_enabled=True, view_mode="first_person",
                                       arm=ArmMappingConfig(smoothing=1.0)))
    ee = pose((0.3, 0.2, 0.6))
    hand = np.zeros((25, 3))
    hand[:, 0] = np.linspace(0.0, 0.2, 25)
    wrist = pose((0.5, 0.0, 1.0))
    f = InputFrame(now=0.0, joys={"left": joy()}, hands={"left": hand},
                   hand_wrists={"left": wrist}, ee_poses={"left": ee})
    out = policy.step(f)
    np.testing.assert_allclose(out.arm_targets["left"], ee)
    assert policy.hand_mode["left"] == "tracking"
    assert matrix_to_quat(out.arm_targets["left"][:3, :3])[0] == pytest.approx(1.0)


TORSO_HOME = pose((-0.05, 0.0, 0.75))
HEAD0 = pose((0.0, 0.0, 1.6))


def _torso_frame(t, head, grip=1.0, stick_y=0.0, torso=TORSO_HOME):
    return InputFrame(
        now=t, headset=head,
        controllers={"left": pose((0.3, 0.2, 1.0)), "right": None},
        joys={"left": joy(grip=grip), "right": joy(stick=(0.0, stick_y))},
        ee_poses={"left": pose((0.4, 0.2, 0.9))},
        torso_pose=torso, torso_home=TORSO_HOME)


def _torso_policy(view_mode="first_person", pitch_source="translation", **torso):
    from lookie_teleop.first_person import FirstPersonConfig
    from lookie_teleop.policy import TorsoFollowConfig
    cfg = TeleopConfig(start_enabled=True, enable_hands=False, view_mode=view_mode,
                       first_person=FirstPersonConfig(pitch_source=pitch_source),
                       torso=TorsoFollowConfig(**{"smoothing": 1.0, **torso}))
    return TeleopPolicy(cfg)


def _run(policy, head, t0=0.0, steps=200, **kw):
    out = None
    for i in range(steps):
        out = policy.step(_torso_frame(t0 + 0.02 * i, head, **kw))
    return out


def _ramp(policy, start, end, t0, steps=50):
    """Move the head smoothly (no jump rejection) then hold."""
    from scipy.spatial.transform import Rotation, Slerp
    rots = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], end[:3, :3]]))
    for i in range(steps):
        a = (i + 1) / steps
        policy.step(_torso_frame(t0 + 0.02 * i, pose((1 - a) * start[:3, 3] + a * end[:3, 3],
                                                      rots([a]).as_matrix()[0])))
    return _run(policy, end, t0=t0 + 0.02 * steps)


def _offsets(target):
    from lookie_teleop.policy import TorsoFollow
    return TorsoFollow.offsets(target, TORSO_HOME)


def test_torso_idle_until_clutch():
    policy = _torso_policy()
    out = _run(policy, pose((0.0, 0.0, 1.4)), grip=0.0, steps=5)
    assert out.torso_target is None and not out.state["torso_follow"]


def test_torso_follows_height_lean_and_yaw():
    policy = _torso_policy()
    _run(policy, HEAD0, steps=2)
    # Squat 15 cm, lean the head 12 cm forward, turn 30 degrees
    head = pose((0.12, 0.0, 1.45), rot_z(math.radians(30)))
    s = _offsets(_ramp(policy, HEAD0, head, t0=1.0).torso_target)
    assert s[1] == pytest.approx(-0.13, abs=1e-3)   # 15 cm - 2 cm deadband
    assert s[0] == pytest.approx(0.0)                # forward_scale 0
    assert s[2] == pytest.approx(math.atan2(0.12, 0.6) - 0.05, abs=1e-3)
    assert s[3] == pytest.approx(math.radians(30) - 0.05, abs=1e-3)


def test_torso_deltas_use_operator_heading():
    # Operator faces +Y in VR: moving the head along +Y is "forward".
    policy = _torso_policy(forward_scale=1.0, pitch_scale=0.0)
    facing = rot_z(math.pi / 2)
    _run(policy, pose((0.0, 0.0, 1.6), facing), steps=2)
    s = _offsets(_ramp(policy, pose((0.0, 0.0, 1.6), facing), pose((0.0, 0.12, 1.6), facing), t0=1.0).torso_target)
    assert s[0] == pytest.approx(0.10, abs=1e-3)
    assert s[3] == pytest.approx(0.0, abs=1e-6)


def test_torso_limits_speed_and_hold_on_release():
    policy = _torso_policy()
    _run(policy, HEAD0, steps=2)
    # Head drops 14 cm in one tick (under jump_reject): the target is speed capped
    z0 = _offsets(policy.step(_torso_frame(1.0, pose((0.0, 0.0, 1.46)))).torso_target)[1]
    out = policy.step(_torso_frame(1.02, pose((0.0, 0.0, 1.46))))
    assert _offsets(out.torso_target)[1] - z0 == pytest.approx(-0.3 * 0.02, abs=1e-6)
    s = _offsets(_ramp(policy, pose((0.0, 0.0, 1.46)), pose((0.0, 0.0, 0.8)), t0=2.0).torso_target)
    assert s[1] == pytest.approx(-0.30)  # height limit
    # Release and stand up: the torso holds
    for i in range(50):
        out = policy.step(_torso_frame(7.0 + 0.02 * i, pose((0.0, 0.0, 0.8 + 0.016 * (i + 1))), grip=0.0))
    s2 = _offsets(out.torso_target)
    assert s2[1] == pytest.approx(-0.30)
    # Re-grip while standing: grip never re-anchors, back to the reference mapping
    s3 = _offsets(_run(policy, HEAD0, t0=10.0).torso_target)
    assert s3[1] == pytest.approx(0.0)


def test_torso_stick_offset_and_home_reset():
    policy = _torso_policy()
    s = _offsets(_run(policy, HEAD0, grip=0.0, stick_y=-1.0, steps=26).torso_target)
    assert s[1] == pytest.approx(0.15 * 0.5, abs=1e-3)  # stick forward = up, 0.5 s
    s = _offsets(_run(policy, HEAD0, t0=1.0, grip=0.0, stick_y=-1.0, steps=100).torso_target)
    assert s[1] == pytest.approx(0.10)  # clamped, no wind-up
    s = _offsets(_run(policy, HEAD0, t0=3.0, grip=0.0, stick_y=1.0, steps=26).torso_target)
    assert s[1] == pytest.approx(0.10 - 0.15 * 0.52, abs=1e-3)
    # B long press (calibrate) stops teleop and clears the torso command
    policy.step(InputFrame(now=10.0, joys={"right": joy(secondary=1)}, torso_home=TORSO_HOME))
    out = policy.step(InputFrame(now=11.1, joys={"right": joy(secondary=1)}, torso_home=TORSO_HOME))
    assert "calibrate" in out.events and out.torso_target is None and not policy.enabled


def test_torso_head_pitch_source():
    policy = _torso_policy(pitch_source="head")
    _run(policy, HEAD0, steps=2)
    c, s = math.cos(0.4), math.sin(0.4)
    look_down = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    out = _ramp(policy, HEAD0, pose((0.0, 0.0, 1.6), look_down), t0=1.0)
    assert _offsets(out.torso_target)[2] == pytest.approx(0.35, abs=1e-3)


@pytest.mark.parametrize("view_mode", ["first_person", "third_person"])
def test_calibration_reference_survives_grip(view_mode):
    """B maps the current controller pose to the EE home; grip never re-anchors."""
    cfg = TeleopConfig(enable_hands=False, view_mode=view_mode)
    cfg.arm.smoothing = 1.0
    policy = TeleopPolicy(cfg)
    home = pose((0.4, 0.2, 0.9))

    def frame(t, ctrl, grip=0.0, a=0, b=0, ee=home):
        return InputFrame(
            now=t, headset=HEAD0,
            controllers={"left": ctrl, "right": None},
            joys={"left": joy(grip=grip), "right": joy(primary=a, secondary=b)},
            ee_poses={"left": ee}, ee_homes={"left": home})

    c0 = pose((0.3, 0.2, 1.0))
    t = 0.0
    events = []
    for _ in range(60):  # right B long press
        events += policy.step(frame(t, c0, b=1)).events
        t += 0.02
    assert "calibrate" in events
    for a in (0, 1, 0):  # right A
        policy.step(frame(t, c0, a=a))
        t += 0.02
    assert policy.enabled

    # Grip with the controller 10 cm off the calibration pose: the target starts
    # at the measured EE and moves (speed capped) to home + 10 cm.
    c1 = pose((0.4, 0.2, 1.0))
    out = policy.step(frame(t, c1, grip=1.0))
    t += 0.02
    assert np.allclose(out.arm_targets["left"][:3, 3], home[:3, 3], atol=1e-9)
    out = policy.step(frame(t, c1, grip=1.0))
    t += 0.02
    assert np.linalg.norm(out.arm_targets["left"][:3, 3] - home[:3, 3]) == pytest.approx(1.5 * 0.02)
    for _ in range(100):
        out = policy.step(frame(t, c1, grip=1.0))
        t += 0.02
    assert np.allclose(out.arm_targets["left"][:3, 3], [0.5, 0.2, 0.9], atol=1e-6)

    # Release, move the controller, re-grip: still the calibrated mapping.
    for _ in range(5):
        policy.step(frame(t, c1))
        t += 0.02
    c2 = pose((0.3, 0.2, 1.05))
    for _ in range(100):
        out = policy.step(frame(t, c2, grip=1.0, ee=pose((0.5, 0.2, 0.9))))
        t += 0.02
    assert np.allclose(out.arm_targets["left"][:3, 3], [0.4, 0.2, 0.95], atol=1e-6)


def test_default_view_mode_is_third_person():
    assert TeleopConfig().view_mode == "third_person"
    with pytest.raises(ValueError):
        TeleopPolicy(TeleopConfig(view_mode="bogus"))


def _calibrated(policy, head, ctrl=None, home=None, t=0.0):
    """Calibrate (B long press) and start (A); returns the next time."""
    ctrl = pose((0.3, 0.2, 1.0)) if ctrl is None else ctrl
    home = pose((0.4, 0.2, 0.9)) if home is None else home

    def f(t, a=0, b=0):
        return InputFrame(now=t, headset=head, controllers={"left": ctrl, "right": None},
                          joys={"left": joy(), "right": joy(primary=a, secondary=b)},
                          ee_poses={"left": home}, ee_homes={"left": home},
                          torso_pose=TORSO_HOME, torso_home=TORSO_HOME)
    for _ in range(60):
        policy.step(f(t, b=1))
        t += 0.02
    for a in (0, 1, 0):
        policy.step(f(t, a=a))
        t += 0.02
    assert policy.enabled
    return t


# Chest-mounted headset hanging forward-down at 30 deg (device x forward-down).
MOUNT = np.array([[math.cos(0.5), 0, math.sin(0.5)], [0, 1, 0], [-math.sin(0.5), 0, math.cos(0.5)]])
CHEST0 = pose((0.0, 0.0, 1.3), MOUNT)
PIVOT = np.array([-0.1, 0.0, -0.45])


def _chest(lean=0.0, yaw=0.0, dz=0.0):
    """Chest pose after leaning ``lean`` about the hip pivot and turning ``yaw``."""
    from lookie_teleop.common import rot_y
    hip = CHEST0[:3, 3] + PIVOT + np.array([0.0, 0.0, dz])
    r_body = rot_z(yaw) @ rot_y(lean)
    return pose(hip - r_body @ PIVOT, r_body @ MOUNT)


def test_third_person_chest_lean_and_turn():
    policy = _torso_policy(view_mode="third_person")
    t = _calibrated(policy, CHEST0)
    # Lean 0.3 rad about the hip and crouch 10 cm: pitch only, no forward motion
    s = _offsets(_ramp(policy, CHEST0, _chest(lean=0.3, dz=-0.10), t0=t).torso_target)
    assert s[2] == pytest.approx(0.3 - 0.05, abs=1e-3)
    assert s[1] == pytest.approx(-0.08, abs=1e-3)
    assert s[3] == pytest.approx(0.0, abs=1e-6)
    # Turn the body 40 deg (upright): torso yaw follows despite the tilted mount
    s = _offsets(_ramp(policy, _chest(lean=0.3, dz=-0.10), _chest(yaw=0.7), t0=t + 10.0).torso_target)
    assert s[3] == pytest.approx(0.7 - 0.05, abs=1e-3)
    assert s[2] == pytest.approx(0.0, abs=1e-3)


def test_third_person_arms_follow_the_body_frame():
    from lookie_teleop.common import rot_y
    cfg = TeleopConfig(enable_hands=False, view_mode="third_person")
    cfg.arm.smoothing = 1.0
    cfg.torso.enabled = False  # isolate the arm mapping (torso frame stays home)
    policy = TeleopPolicy(cfg)
    home = pose((0.4, 0.2, 0.9))
    c0 = pose((0.3, 0.2, 1.0))
    t = _calibrated(policy, CHEST0, ctrl=c0, home=home)

    def f(t, head, ctrl):
        return InputFrame(now=t, headset=head, controllers={"left": ctrl, "right": None},
                          joys={"left": joy(grip=1.0), "right": joy()},
                          ee_poses={"left": home}, ee_homes={"left": home})
    # Turn the whole body (chest + controller) by 90 deg: arm target unchanged
    turn = rot_z(math.pi / 2)
    body = pose(r=turn)
    out = None
    for i in range(100):
        out = policy.step(f(t, body @ CHEST0, body @ c0))
        t += 0.02
    np.testing.assert_allclose(out.arm_targets["left"], home, atol=1e-6)
    # Reaching 10 cm forward in the turned body (world +Y) is robot +X
    reach = pose((0.0, 0.1, 0.0)) @ body @ c0
    for i in range(100):
        out = policy.step(f(t, body @ CHEST0, reach))
        t += 0.02
    np.testing.assert_allclose(out.arm_targets["left"][:3, 3], [0.5, 0.2, 0.9], atol=1e-6)
    # Losing the headset holds the target
    held = out.arm_targets["left"].copy()
    out = policy.step(f(t, None, pose((1.0, 1.0, 1.0))))
    np.testing.assert_allclose(out.arm_targets["left"], held)


def test_third_person_arm_target_turns_with_torso():
    from lookie_teleop.common import MappingContext, torso_frame
    from lookie_teleop.third_person import BodyArmMapping
    m = BodyArmMapping(ArmMappingConfig())
    home = pose((0.4, 0.2, 0.9))
    f0 = torso_frame(np.zeros(4), TORSO_HOME)
    m.set_reference(pose((0.3, 0.2, 1.0)), home, MappingContext(headset=CHEST0, torso_frame=f0))
    m.begin(pose((0.3, 0.2, 1.0)), MappingContext(headset=CHEST0))
    f1 = torso_frame(np.array([0.0, 0.0, 0.0, math.pi / 2]), TORSO_HOME)
    e = m.desired(pose((0.3, 0.2, 1.0)), MappingContext(headset=CHEST0, torso_frame=f1))
    # EE swings about the torso axis with the torso yaw
    rel = home[:3, 3] - TORSO_HOME[:3, 3]
    np.testing.assert_allclose(e[:3, 3], TORSO_HOME[:3, 3] + rot_z(math.pi / 2) @ rel, atol=1e-9)


# --- hand-tracking gesture control -------------------------------------------

from lookie_teleop.gesture import GestureConfig, GestureControl, palm_facing, palm_normal  # noqa: E402

G_HEAD = pose((0.0, 0.0, 1.6))


def make_hand(side, centre, toward=True, fist=False, bend_deg=0.0):
    """Open hand, fingers up, palm towards -x (the headset at the origin) or away."""
    sign = 1.0 if side == "right" else -1.0
    pts = np.zeros((25, 3))
    for k, y in zip(range(5, 25, 5), (-0.03, -0.01, 0.01, 0.03)):
        y *= sign
        pts[k] = (0.0, y * 0.3, 0.01)
        p, angle = np.array([0.0, y, 0.08]), 0.0
        pts[k + 1] = p
        for j in range(3):  # bones bend towards the palm (+x) by 70 deg each
            angle += math.radians(70.0 if fist else bend_deg)
            p = p + 0.03 * np.array([math.sin(angle), 0.0, math.cos(angle)])
            pts[k + 2 + j] = p
    pts[1:5] = [(0.0, -0.03 * sign, 0.02 + 0.02 * i) for i in range(4)]
    if not toward:
        pts[:, :2] *= -1.0
    return pts + np.asarray(centre)


def test_palm_facing_detection():
    cfg = GestureConfig()
    for side, y in (("right", -0.1), ("left", 0.1)):
        c = (0.35, y, 1.45)
        np.testing.assert_allclose(palm_normal(make_hand(side, c), side), [-1, 0, 0], atol=1e-9)
        assert palm_facing(make_hand(side, c), side, G_HEAD, cfg)
        assert not palm_facing(make_hand(side, c, toward=False), side, G_HEAD, cfg)
        assert not palm_facing(make_hand(side, c, fist=True), side, G_HEAD, cfg)


def test_gesture_short_long_and_rearm():
    g = GestureControl(GestureConfig())
    up = {"left": make_hand("left", (0.35, 0.1, 1.45)), "right": make_hand("right", (0.35, -0.1, 1.45))}
    wr = {s: pose(h[0]) for s, h in up.items()}
    t, events = 0.0, []
    for _ in range(30):  # 0.6 s held, then a 0.1 s tracking flicker
        events.append(g.update(up, wr, G_HEAD, t)[1]); t += 0.02
    for _ in range(5):
        events.append(g.update({}, {}, G_HEAD, t)[1]); t += 0.02
    assert g.stage == "hold"
    for _ in range(30):  # 1.3 s in total
        events.append(g.update(up, wr, G_HEAD, t)[1]); t += 0.02
    assert set(events) == {None} and g.stage == "ready"
    for _ in range(20):  # released past release_s
        events.append(g.update({}, {}, G_HEAD, t)[1]); t += 0.02
    assert events.count("short") == 1 and g.stage is None
    # Re-held immediately: ignored until released for rearm_s
    for _ in range(200):
        assert g.update(up, wr, G_HEAD, t)[1] is None; t += 0.02
    for _ in range(30):
        g.update({}, {}, G_HEAD, t); t += 0.02
    events = []
    for _ in range(160):
        events.append(g.update(up, wr, G_HEAD, t)[1]); t += 0.02
    assert events.count("long") == 1
    assert g.update({}, {}, G_HEAD, t)[1] is None


EE_HOME = {"left": pose((0.3, 0.25, 0.9)), "right": pose((0.3, -0.25, 0.9))}


class HandOperator:
    def __init__(self, policy):
        self.policy, self.t, self.events, self.out = policy, 0.0, [], None
        self.ee = dict(EE_HOME)

    def run(self, seconds, hands, dt=0.02):
        for _ in range(int(round(seconds / dt))):
            wrists = {s: pose(h[0]) for s, h in hands.items()}
            self.out = self.policy.step(InputFrame(
                now=self.t, headset=G_HEAD, hands=dict(hands), hand_wrists=wrists,
                ee_poses=dict(self.ee), ee_homes=dict(EE_HOME)))
            self.events += self.out.events
            self.t += dt
        return self.out


GESTURE = {"left": make_hand("left", (0.35, 0.1, 1.45)), "right": make_hand("right", (0.35, -0.1, 1.45))}
WORK = {"left": make_hand("left", (0.35, 0.2, 1.1), toward=False),
        "right": make_hand("right", (0.35, -0.2, 1.1), toward=False)}


def _shift(hands, d):
    return {s: h + np.asarray(d) for s, h in hands.items()}


@pytest.mark.parametrize("view_mode", ["first_person", "third_person"])
def test_gesture_calibrate_start_stop(view_mode):
    policy = TeleopPolicy(TeleopConfig(view_mode=view_mode, arm=ArmMappingConfig(smoothing=1.0)))
    op = HandOperator(policy)
    op.run(1.6, GESTURE)
    op.run(0.4, WORK)
    assert op.events == ["calibrate"] and policy._awaiting_reference
    assert not policy.calibrated
    op.run(1.6, WORK)  # settle at home posture -> reference
    assert op.events == ["calibrate", "reference"] and policy.calibrated
    op.run(0.5, WORK)
    assert not op.out.arm_targets and not op.out.hand_positions

    op.run(1.6, GESTURE)
    out = op.run(0.4, WORK)
    assert op.events[-1] == "enabled" and policy.enabled
    # Wrists are back at the calibration posture: aligned, engaged
    np.testing.assert_allclose(out.arm_targets["right"], EE_HOME["right"], atol=1e-9)
    assert set(out.hand_positions) == {"left", "right"}

    # Short hold while running: arms keep following, no stop
    op.run(0.6, WORK)
    op.run(1.6, GESTURE)
    out = op.run(0.4, WORK)
    assert op.events[-1] == "enabled" and policy.enabled
    assert set(out.arm_targets) == {"left", "right"}

    # Long hold recalibrates
    op.run(0.6, WORK)
    op.run(3.1, GESTURE)
    assert op.events[-1] == "calibrate" and not policy.calibrated


def _running(view_mode="first_person"):
    policy = TeleopPolicy(TeleopConfig(view_mode=view_mode, arm=ArmMappingConfig(smoothing=1.0)))
    op = HandOperator(policy)
    op.run(1.6, GESTURE); op.run(2.0, WORK)
    op.run(1.6, GESTURE); op.run(0.4, WORK)
    assert policy.enabled
    return policy, op


def test_hand_engages_only_when_aligned():
    policy, op = _running()
    op.run(1.0, WORK)
    away = _shift(WORK, (0.0, 0.0, 0.15))
    out = op.run(0.06, away)  # tracking jump while engaged is absorbed, keeps following
    policy.disengage_all(block=False)
    out = op.run(0.2, away)
    assert not out.arm_targets and not out.hand_positions
    out = op.run(0.2, _shift(WORK, (0.0, 0.0, 0.03)))
    assert set(out.arm_targets) == {"left", "right"}


def test_hand_tracking_dropout_holds_then_releases():
    policy, op = _running()
    out = op.run(0.2, WORK)
    held = out.arm_targets["left"].copy()
    out = op.run(0.1, {"right": WORK["right"]})
    np.testing.assert_allclose(out.arm_targets["left"], held)
    out = op.run(0.2, {"right": WORK["right"]})
    assert "left" not in out.arm_targets and policy.arms["left"].engaged is False
    assert "right" in out.arm_targets


def test_gesture_needs_stillness_for_long_hold():
    g = GestureControl(GestureConfig())
    t = 0.0
    # Palms toward the headset but drifting slowly (below still_speed): never fires
    for i in range(250):
        d = (0.0, 0.0, 0.002 * (i % 125) - 0.12)  # 0.1 m/s sweep
        h = _shift(GESTURE, d)
        ev = g.update(h, {s: pose(x[0]) for s, x in h.items()}, G_HEAD, t)[1]
        assert ev is None
        t += 0.02
    # Held still: long fires
    events = []
    for _ in range(160):
        events.append(g.update(GESTURE, {s: pose(x[0]) for s, x in GESTURE.items()}, G_HEAD, t)[1])
        t += 0.02
    assert events.count("long") == 1


def test_gesture_needs_still_fingers_for_long_hold():
    g = GestureControl(GestureConfig())
    t = 0.0
    # Wrists still and palms open toward the headset, but fingers flexing: never fires
    for i in range(250):
        b = 15.0 if (i // 10) % 2 else 0.0
        h = {s: make_hand(s, c, bend_deg=b) for s, c in
             (("left", (0.35, 0.1, 1.45)), ("right", (0.35, -0.1, 1.45)))}
        ev = g.update(h, {s: pose(x[0]) for s, x in h.items()}, G_HEAD, t)[1]
        assert ev is None
        t += 0.02
    events = []
    for _ in range(160):
        events.append(g.update(GESTURE, {s: pose(x[0]) for s, x in GESTURE.items()}, G_HEAD, t)[1])
        t += 0.02
    assert events.count("long") == 1


def test_palm_toward_headset_while_running_keeps_following():
    policy, op = _running()
    op.run(0.5, WORK)
    out = op.run(1.0, GESTURE)  # hand moved up and turned: not a stop, still following
    assert policy.enabled and set(out.arm_targets) == {"left", "right"}
