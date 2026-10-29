"""ROS-free teleoperation policy for Lookie.

The policy turns one snapshot of XR input into task-space commands:

* arm EE targets (relative "clutch" mapping, in the robot base frame),
* Revo2 hand joint positions,
* a chassis velocity command,
* discrete events (home requests).

All poses are 4x4 homogeneous matrices. XR poses are in the xr_media ``world``
frame (Z-up, +X forward, +Y left); robot poses are in the base frame.
"""

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .gesture import GestureConfig, GestureControl, aligned
from .hand_retarget import HandRetargetConfig, HandRetargeter, retarget_trigger
from .common import MappingContext, mount_rotation, rot_y, torso_frame
from .first_person import FirstPersonConfig, HeadTorsoSource, WorldArmMapping
from .third_person import BodyArmMapping, ChestTorsoSource, ThirdPersonConfig
from .transforms import matrix_to_quat, quat_to_matrix, rot_z, slerp, yaw_of

VIEW_MODES = ("first_person", "third_person")

SIDES = ("left", "right")

# xr_media Joy layout
AXIS_STICK_X, AXIS_STICK_Y, AXIS_TRIGGER, AXIS_GRIP = 0, 1, 2, 3
BUTTON_TRIGGER, BUTTON_GRIP, BUTTON_STICK, BUTTON_PRIMARY, BUTTON_SECONDARY = 0, 1, 2, 3, 4


def apply_deadzone(value: float, deadzone: float) -> float:
    deadzone = max(0.0, min(0.95, float(deadzone)))
    value = max(-1.0, min(1.0, float(value)))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


def _axis(values: Sequence[float], index: int) -> float:
    return float(values[index]) if 0 <= index < len(values) else 0.0


def _button(values: Sequence[int], index: int) -> bool:
    return bool(values[index]) if 0 <= index < len(values) else False


class ButtonPress:
    """Classify a button into short (on release) and long (while held) presses."""

    def __init__(self, long_press_s: float = 1.0) -> None:
        self.long_press_s = float(long_press_s)
        self._pressed_at: float | None = None
        self._long_fired = False

    def reset(self) -> None:
        self._pressed_at, self._long_fired = None, False

    def update(self, pressed: bool, now: float) -> str | None:
        if pressed:
            if self._pressed_at is None:
                self._pressed_at, self._long_fired = now, False
            elif not self._long_fired and now - self._pressed_at >= self.long_press_s:
                self._long_fired = True
                return "long"
            return None
        if self._pressed_at is None:
            return None
        fired, self._pressed_at = self._long_fired, None
        return None if fired else "short"


class ClutchGate:
    """Grip clutch with hysteresis; optional double-press-then-hold arming.

    ``mode='hold'``: engaged while grip >= engage threshold, released below the
    release threshold. ``mode='double_press'``: a quick full tap, release, then
    a second press that is held (avoids accidental engagement).
    """

    def __init__(
        self,
        mode: str = "hold",
        engage_threshold: float = 0.6,
        release_threshold: float = 0.3,
        tap_max_s: float = 0.35,
        second_press_timeout_s: float = 0.4,
    ) -> None:
        if mode not in ("hold", "double_press"):
            raise ValueError(f"unknown clutch mode '{mode}'")
        self.mode = mode
        self.engage = float(engage_threshold)
        self.release = float(release_threshold)
        self.tap_max_s = float(tap_max_s)
        self.second_timeout_s = float(second_press_timeout_s)
        self.reset()

    def reset(self) -> None:
        self._state = "idle"
        self._t = 0.0
        self.latched_off = False

    @property
    def active(self) -> bool:
        return self._state == "active"

    def block_until_release(self) -> None:
        """Disengage and ignore the grip until it is released."""
        self.reset()
        self.latched_off = True

    def update(self, value: float, now: float) -> bool:
        pressed = value >= self.engage
        released = value <= self.release
        if self.latched_off:
            if released:
                self.latched_off = False
            return False
        if self.mode == "hold":
            if self._state == "active":
                if released:
                    self._state = "idle"
            elif pressed:
                self._state = "active"
            return self.active
        # double_press
        if self._state == "active":
            if released:
                self._state = "idle"
        elif self._state == "idle":
            if pressed:
                self._state, self._t = "first", now
        elif self._state == "first":
            if released:
                self._state = "armed" if now - self._t <= self.tap_max_s else "idle"
                self._t = now
        elif self._state == "armed":
            if pressed:
                self._state = "active" if now - self._t <= self.second_timeout_s else "first"
                self._t = now
            elif now - self._t > self.second_timeout_s:
                self._state = "idle"
        return self.active


@dataclass
class ArmMappingConfig:
    position_scale: float = 1.0
    rotation_enabled: bool = True
    max_offset_m: float = 1.0
    # Workspace box in the base frame: [xmin, ymin, zmin], [xmax, ymax, zmax]
    workspace_min: tuple[float, float, float] = (-0.2, -1.0, 0.0)
    workspace_max: tuple[float, float, float] = (1.2, 1.0, 1.8)
    # Low-pass factor per tick in (0, 1]; 1 disables filtering.
    smoothing: float = 0.5
    # Target speed caps (m/s, rad/s; 0 = off). They bound how hard a fast
    # swing can drive the IK; keep them above normal operator speeds.
    max_linear_speed: float = 1.5
    max_angular_speed: float = 6.0
    # A controller position change larger than this within one update is a
    # tracking glitch (lost/re-acquired pose): the clutch re-anchors instead
    # of following it (0 = off).
    jump_reject_m: float = 0.15


class ArmClutch:
    """Pose mapping relative to a calibration reference.

    The reference (controller pose, robot EE pose, operator heading / body
    frame) is set at calibration and kept across clutch cycles, so re-gripping
    returns the arm to the calibrated mapping (speed capped) instead of
    re-anchoring. Without a reference, the first engage latches one. The pose
    math lives in ``mapping`` (:class:`~.first_person.WorldArmMapping` by
    default, :class:`~.third_person.BodyArmMapping` for a chest headset);
    this class adds workspace clipping, smoothing and speed caps.
    """

    def __init__(self, config: ArmMappingConfig, mapping=None) -> None:
        self.config = config
        self.mapping = mapping if mapping is not None else WorldArmMapping(config)
        self.reset()

    def reset(self) -> None:
        """Release and drop the reference."""
        self.mapping.clear()
        self.release()

    def release(self) -> None:
        self.engaged = False
        self._last_now: float | None = None
        self.target: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self.mapping.has_reference

    def set_reference(self, controller: np.ndarray, ee: np.ndarray, ctx) -> None:
        self.mapping.set_reference(controller, ee, _context(ctx))

    def engage(self, controller: np.ndarray, ee: np.ndarray, ctx,
               now: float | None = None) -> None:
        """Start following from the measured EE pose ``ee``.

        ``ctx`` is a :class:`MappingContext` or just the operator yaw.
        """
        ctx = _context(ctx)
        if not self.has_reference:
            self.mapping.set_reference(controller, ee, ctx)
        self.mapping.begin(controller, ctx)
        self.engaged = True
        self._last_now = now
        self.target = np.asarray(ee, dtype=float).copy()

    def update(self, controller: np.ndarray, now: float | None = None,
               ctx: MappingContext | None = None) -> np.ndarray:
        """Return the filtered EE target; ``now`` (s) enables the speed caps."""
        assert self.engaged
        cfg = self.config
        ctx = ctx if ctx is not None else MappingContext()
        desired = None
        if not self.mapping.has_reference:  # e.g. headset was missing at engage
            self.mapping.set_reference(controller, self.target, ctx)
            self.mapping.begin(controller, ctx)
        if self.mapping.has_reference:
            desired = self.mapping.desired(np.asarray(controller, dtype=float), ctx)
        if desired is None:  # mapping input missing: hold
            if now is not None:
                self._last_now = now
            return self.target
        desired[:3, 3] = np.clip(desired[:3, 3], cfg.workspace_min, cfg.workspace_max)

        alpha = float(np.clip(cfg.smoothing, 1e-3, 1.0))
        previous = self.target if self.target is not None else desired
        filtered = np.eye(4)
        filtered[:3, 3] = previous[:3, 3] + alpha * (desired[:3, 3] - previous[:3, 3])
        q_prev = matrix_to_quat(previous[:3, :3])
        q_new = slerp(q_prev, matrix_to_quat(desired[:3, :3]), alpha)

        dt = None if now is None or self._last_now is None else now - self._last_now
        if now is not None:
            self._last_now = now
        if dt is not None:
            dt = min(max(dt, 0.0), 0.1)
            step = filtered[:3, 3] - previous[:3, 3]
            limit = cfg.max_linear_speed * dt
            if cfg.max_linear_speed > 0.0 and float(np.linalg.norm(step)) > limit:
                filtered[:3, 3] = previous[:3, 3] + step * (limit / float(np.linalg.norm(step)))
            angle = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_prev, q_new)))))
            limit = cfg.max_angular_speed * dt
            if cfg.max_angular_speed > 0.0 and angle > limit:
                q_new = slerp(q_prev, q_new, limit / angle if angle > 0.0 else 0.0)
        filtered[:3, :3] = quat_to_matrix(q_new)
        self.target = filtered
        return filtered


@dataclass
class ChassisConfig:
    enabled: bool = True
    deadzone: float = 0.15
    max_linear: float = 0.4
    max_angular: float = 0.8
    # WebXR sticks: right = +x, forward = -y. Robot: +x forward, +y left, +z CCW.
    forward_sign: float = -1.0
    lateral_sign: float = -1.0
    yaw_sign: float = -1.0
    linear_side: str = "left"
    angular_side: str = "right"


def map_chassis(axes: dict[str, Sequence[float] | None], config: ChassisConfig) -> tuple[float, float, float]:
    if not config.enabled:
        return 0.0, 0.0, 0.0
    lin = axes.get(config.linear_side) or ()
    ang = axes.get(config.angular_side) or ()
    return (
        config.forward_sign * config.max_linear * apply_deadzone(_axis(lin, AXIS_STICK_Y), config.deadzone),
        config.lateral_sign * config.max_linear * apply_deadzone(_axis(lin, AXIS_STICK_X), config.deadzone),
        config.yaw_sign * config.max_angular * apply_deadzone(_axis(ang, AXIS_STICK_X), config.deadzone),
    )


def _soft_deadband(value: float, band: float) -> float:
    if abs(value) <= band:
        return 0.0
    return math.copysign(abs(value) - band, value)


def _context(ctx) -> MappingContext:
    return ctx if isinstance(ctx, MappingContext) else MappingContext(operator_yaw=float(ctx))


@dataclass
class TorsoFollowConfig:
    """Headset -> torso mapping (height, forward, pitch, yaw), Astribot-style.

    Deltas come from a view-mode source (head-worn or chest-mounted headset)
    relative to the pose recorded at calibration (or at the first engage
    without one). Scales of 0 turn an axis off. Limits are offsets from the
    torso home pose.
    """

    enabled: bool = True
    height_scale: float = 1.0
    forward_scale: float = 0.0
    pitch_scale: float = 1.0
    yaw_scale: float = 1.0
    # Deadbands (m, m, rad, rad) on the headset deltas, removing standing sway
    height_deadband_m: float = 0.02
    forward_deadband_m: float = 0.02
    pitch_deadband_rad: float = 0.05
    yaw_deadband_rad: float = 0.05
    height_limits_m: tuple[float, float] = (-0.30, 0.10)
    forward_limits_m: tuple[float, float] = (-0.05, 0.15)
    pitch_limits_rad: tuple[float, float] = (-0.15, 0.6)
    yaw_limits_rad: tuple[float, float] = (-1.2, 1.2)
    smoothing: float = 0.3
    max_linear_speed: float = 0.3
    max_angular_speed: float = 1.0
    jump_reject_m: float = 0.15
    # Stick height offset, added on top of the headset follow at any time
    stick_side: str = "right"
    stick_rate_mps: float = 0.15
    stick_sign: float = -1.0  # WebXR stick forward = -y -> torso up
    stick_deadzone: float = 0.3


class TorsoFollow:
    """Torso target as 4 offsets from home: [x, z, pitch, yaw] in the base frame.

    Pose = (p_home + [x, 0, z], Rz(yaw) Ry(pitch) R_home), i.e. lean then twist,
    so a lean after a turn goes along the turned direction. ``source`` turns
    headset poses into raw deltas (:class:`~.first_person.HeadTorsoSource` or
    :class:`~.third_person.ChestTorsoSource`).
    """

    def __init__(self, config: TorsoFollowConfig, source=None) -> None:
        self.config = config
        self.source = source if source is not None else HeadTorsoSource(FirstPersonConfig(), (1.0, 0.0, 0.0))
        self.state: np.ndarray | None = None  # filtered command
        self.reset()

    def reset(self) -> None:
        """Forget the command and the calibration reference."""
        self.engaged = False
        self.state = None
        self.stick_offset = 0.0
        self._s0 = np.zeros(4)      # offsets that map to the reference head pose
        self._held: np.ndarray | None = None  # held offsets (minus stick) while released
        self._last_now: float | None = None
        self.source.clear()

    @property
    def has_reference(self) -> bool:
        return self.source.has_reference

    def set_reference(self, head: np.ndarray, ctx, offsets: np.ndarray | None = None) -> None:
        """Headset pose ``head`` maps to torso ``offsets`` (default: home)."""
        self.source.set_reference(np.asarray(head, dtype=float), _context(ctx))
        self._s0 = np.zeros(4) if offsets is None else np.asarray(offsets, dtype=float).copy()

    @staticmethod
    def offsets(pose: np.ndarray, home: np.ndarray) -> np.ndarray:
        rel = pose[:3, :3] @ home[:3, :3].T
        yaw = math.atan2(rel[1, 0], rel[0, 0])
        pitch = math.atan2(-rel[2, 0], math.hypot(rel[0, 0], rel[1, 0]))
        d = pose[:3, 3] - home[:3, 3]
        return np.array([d[0], d[2], pitch, yaw])

    @staticmethod
    def compose(s: np.ndarray, home: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, 3] = home[:3, 3] + np.array([s[0], 0.0, s[1]])
        out[:3, :3] = rot_z(s[3]) @ rot_y(s[2]) @ home[:3, :3]
        return out

    def _limits(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        lo = np.array([c.forward_limits_m[0], c.height_limits_m[0], c.pitch_limits_rad[0], c.yaw_limits_rad[0]])
        hi = np.array([c.forward_limits_m[1], c.height_limits_m[1], c.pitch_limits_rad[1], c.yaw_limits_rad[1]])
        return lo, hi

    def engage(self, head: np.ndarray, torso: np.ndarray, home: np.ndarray, ctx) -> None:
        self.engaged = True
        # Rate limiting starts from the last command (or the measurement).
        if self.state is None:
            self.state = self.offsets(torso, home)
        if not self.has_reference:
            s0 = self.state.copy()
            s0[1] -= self.stick_offset
            self.set_reference(head, ctx, s0)
        self.source.begin(np.asarray(head, dtype=float))

    def release(self) -> None:
        if self.engaged and self.state is not None:
            self._held = self.state.copy()
            self._held[1] -= self.stick_offset
        self.engaged = False

    def update(self, now: float, head: np.ndarray | None, torso: np.ndarray | None,
               home: np.ndarray, stick: float) -> np.ndarray | None:
        """Return the torso target (base frame) or ``None`` if never commanded."""
        c = self.config
        dt = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), 0.1)
        self._last_now = now
        stick = apply_deadzone(stick, c.stick_deadzone)
        lo, hi = self._limits()
        if stick != 0.0:
            if self.state is None:
                if torso is None:
                    return None
                self.state = self.offsets(torso, home)
                self._held = self.state - np.array([0.0, self.stick_offset, 0.0, 0.0])
            self.stick_offset += c.stick_sign * c.stick_rate_mps * stick * dt
        if self.state is None:
            return None

        if self.engaged:
            desired = self._s0.copy()
        elif self._held is not None:
            desired = self._held.copy()
        else:
            desired = self.state - np.array([0.0, self.stick_offset, 0.0, 0.0])
        # Keep the stick offset inside the height range so it never winds up
        # (against the base offset only, so head motion is never absorbed).
        self.stick_offset = float(np.clip(desired[1] + self.stick_offset, lo[1], hi[1]) - desired[1])
        if self.engaged and head is not None and self.has_reference:
            dx, dz, pitch, yaw = self.source.deltas(np.asarray(head, dtype=float), c.jump_reject_m)
            desired += np.array([
                c.forward_scale * _soft_deadband(dx, c.forward_deadband_m),
                c.height_scale * _soft_deadband(dz, c.height_deadband_m),
                c.pitch_scale * _soft_deadband(pitch, c.pitch_deadband_rad),
                c.yaw_scale * _soft_deadband(yaw, c.yaw_deadband_rad),
            ])
        desired[1] += self.stick_offset
        desired = np.clip(desired, lo, hi)

        alpha = float(np.clip(c.smoothing, 1e-3, 1.0))
        step = alpha * (desired - self.state)
        speeds = np.array([c.max_linear_speed, c.max_linear_speed, c.max_angular_speed, c.max_angular_speed])
        step = np.where(speeds > 0.0, np.clip(step, -speeds * dt, speeds * dt), step)
        self.state = self.state + step
        return self.compose(self.state, home)


@dataclass
class Joy:
    axes: Sequence[float]
    buttons: Sequence[int]


@dataclass
class InputFrame:
    """Fresh inputs only; stale inputs must be passed as ``None``."""

    now: float
    headset: np.ndarray | None = None
    controllers: dict[str, np.ndarray | None] = field(default_factory=dict)
    joys: dict[str, Joy | None] = field(default_factory=dict)
    hands: dict[str, np.ndarray | None] = field(default_factory=dict)  # 25x3 joint positions
    hand_wrists: dict[str, np.ndarray | None] = field(default_factory=dict)  # 4x4 wrist poses
    ee_poses: dict[str, np.ndarray | None] = field(default_factory=dict)  # base frame
    ee_homes: dict[str, np.ndarray | None] = field(default_factory=dict)  # base frame
    torso_pose: np.ndarray | None = None  # base frame
    torso_home: np.ndarray | None = None  # base frame


@dataclass
class OutputFrame:
    arm_targets: dict[str, np.ndarray] = field(default_factory=dict)
    hand_positions: dict[str, list[float]] = field(default_factory=dict)
    cmd_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torso_target: np.ndarray | None = None
    events: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)


@dataclass
class TeleopConfig:
    arm: ArmMappingConfig = field(default_factory=ArmMappingConfig)
    hand: HandRetargetConfig = field(default_factory=HandRetargetConfig)
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    torso: TorsoFollowConfig = field(default_factory=TorsoFollowConfig)
    # "third_person": headset hung on the chest, operator watches the robot;
    # "first_person": headset worn on the head.
    view_mode: str = "third_person"
    first_person: FirstPersonConfig = field(default_factory=FirstPersonConfig)
    third_person: ThirdPersonConfig = field(default_factory=ThirdPersonConfig)
    enabled_sides: tuple[str, ...] = SIDES
    clutch_mode: str = "hold"
    clutch_engage_threshold: float = 0.6
    clutch_release_threshold: float = 0.3
    # Hand tracking without controllers: palm gesture and align-to-engage.
    gesture: GestureConfig = field(default_factory=GestureConfig)
    long_press_s: float = 1.0
    start_enabled: bool = False
    # Local headset axis that points where the operator looks.
    headset_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    use_headset_yaw: bool = True
    enable_hands: bool = True


class TeleopPolicy:
    """Stateful mapping from :class:`InputFrame` to :class:`OutputFrame`.

    Buttons (xr_media layout, Astribot-style):
      * right B (secondary) long press : calibrate -- stop teleop, home arms
        and torso (``calibrate``) and take the current controller/headset poses
        as the reference for the home poses. Grip never re-anchors.
      * right A (primary) short press : start teleop; only after a calibration,
        ignored while running (stop with B)
      * grip : arm clutch;  trigger : hand close (controller mode)
      * left stick : chassis translation; right stick X : chassis yaw
      * right stick Y : torso height offset
      * headset : torso height/forward/pitch/yaw while any arm clutch is engaged
      * hand tracking (no controllers): both palms facing the headset, see
        :mod:`.gesture`; an arm engages once the wrist aligns with the EE

    ``view_mode`` selects how the headset is worn (see ``first_person`` /
    ``third_person`` modules): it changes the torso source and whether arms
    are mapped in the world or relative to the operator's chest.
    """

    def __init__(self, config: TeleopConfig | None = None) -> None:
        self.config = config or TeleopConfig()
        cfg = self.config
        self.enabled = bool(cfg.start_enabled)
        self.calibrated = False
        if cfg.view_mode not in VIEW_MODES:
            raise ValueError(f"unknown view_mode '{cfg.view_mode}'")
        self.third_person = cfg.view_mode == "third_person"
        if self.third_person:
            self.arms = {s: ArmClutch(cfg.arm, BodyArmMapping(cfg.arm)) for s in cfg.enabled_sides}
            source = ChestTorsoSource(cfg.third_person)
        else:
            self.arms = {s: ArmClutch(cfg.arm) for s in cfg.enabled_sides}
            source = HeadTorsoSource(cfg.first_person, cfg.headset_forward_axis)
        self._mount: np.ndarray | None = None
        self._torso_home: np.ndarray | None = None
        self.clutches = {
            s: ClutchGate(cfg.clutch_mode, cfg.clutch_engage_threshold, cfg.clutch_release_threshold)
            for s in cfg.enabled_sides
        }
        self.primary = {s: ButtonPress(cfg.long_press_s) for s in SIDES}
        self.secondary = {s: ButtonPress(cfg.long_press_s) for s in SIDES}
        self.hand_mode = {s: "none" for s in SIDES}
        self.hand_retargeters = {s: HandRetargeter(cfg.hand) for s in SIDES}
        self.torso = TorsoFollow(cfg.torso, source)
        self.gesture = GestureControl(cfg.gesture)
        self._gesture_active = False
        self._awaiting_reference = False
        self._lost_since: dict[str, float | None] = {s: None for s in SIDES}

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.calibrated = False
        if not self.enabled:
            self.disengage_all(block=False)
            self.torso.release()

    def calibrate(self, frame: InputFrame) -> None:
        """Stop teleop and map the current operator pose to the home poses.

        The robot itself is homed by the caller. A side without a tracked
        source or home pose falls back to latching at its first engage.
        """
        self.set_enabled(False)
        self._awaiting_reference = False
        yaw = self._operator_yaw(frame)
        if frame.torso_home is not None:
            self._torso_home = frame.torso_home
        self._mount = None
        if self.third_person and frame.headset is not None:
            self._mount = mount_rotation(frame.headset, yaw)
        ctx = MappingContext(operator_yaw=yaw, headset=frame.headset,
                             torso_frame=torso_frame(np.zeros(4), self._torso_home),
                             mount=self._mount)
        for side, arm in self.arms.items():
            arm.reset()
            source = frame.controllers.get(side)
            if source is None:
                source = frame.hand_wrists.get(side)
            home = frame.ee_homes.get(side)
            if source is not None and home is not None:
                arm.set_reference(source, home, ctx)
        for retargeter in self.hand_retargeters.values():
            retargeter.reset()
        self.torso.reset()
        if frame.headset is not None:
            self.torso.set_reference(frame.headset, ctx)
        self.calibrated = True

    def disengage_all(self, block: bool = True) -> None:
        for side in self.arms:
            self.disengage(side, block)

    def disengage(self, side: str, block: bool = True) -> None:
        if side in self.arms:
            self.arms[side].release()
            if block:
                self.clutches[side].block_until_release()
            else:
                self.clutches[side].reset()

    def _operator_yaw(self, frame: InputFrame) -> float:
        if not self.config.use_headset_yaw or frame.headset is None:
            return 0.0
        return yaw_of(frame.headset[:3, :3], self.config.headset_forward_axis)

    def _context(self, frame: InputFrame) -> MappingContext:
        """Mapping inputs for this tick. The torso frame is last tick's command
        (arms step before the torso), else the measured torso pose."""
        if frame.torso_home is not None:
            self._torso_home = frame.torso_home
        home = self._torso_home
        state = self.torso.state
        if state is None and frame.torso_pose is not None and home is not None:
            state = TorsoFollow.offsets(frame.torso_pose, home)
        return MappingContext(
            operator_yaw=self._operator_yaw(frame), headset=frame.headset,
            torso_frame=torso_frame(state if state is not None else np.zeros(4), home),
            mount=self._mount)

    def step(self, frame: InputFrame) -> OutputFrame:
        out = OutputFrame()
        now = frame.now

        for side in SIDES:
            joy = frame.joys.get(side)
            buttons = joy.buttons if joy is not None else ()
            primary = self.primary[side].update(_button(buttons, BUTTON_PRIMARY), now)
            secondary = self.secondary[side].update(_button(buttons, BUTTON_SECONDARY), now)
            if side != "right":
                continue
            if secondary == "long":
                self.calibrate(frame)
                out.events.append("calibrate")
            elif primary == "short" and not self.enabled:
                if self.calibrated:
                    self.set_enabled(True)
                    out.events.append("enabled")
                else:
                    out.events.append("calibration_required")

        self._step_gesture(frame, out)

        axes = {s: (j.axes if j is not None else None) for s, j in frame.joys.items()}
        if self.enabled:
            out.cmd_vel = map_chassis(axes, self.config.chassis)

        ctx = self._context(frame)
        for side in self.arms:
            self._step_arm(side, frame, out, ctx)
        for side in SIDES:
            self._step_hand(side, frame, out)
        self._step_torso(frame, axes, out)

        out.state = {
            "enabled": self.enabled,
            "calibrated": self.calibrated,
            "engaged": {s: a.engaged for s, a in self.arms.items()},
            "hand_mode": dict(self.hand_mode),
            "cmd_vel": list(out.cmd_vel),
            "torso_follow": self.torso.engaged,
            "view_mode": self.config.view_mode,
            "gesture": self._gesture_active,
            # While running only a long hold matters: report it after hold_s, never "ready".
            "gesture_stage": (("hold" if self.gesture.stage == "ready" else None)
                              if self.enabled else self.gesture.stage),
            "awaiting_reference": self._awaiting_reference,
        }
        return out

    def _step_gesture(self, frame: InputFrame, out: OutputFrame) -> None:
        """Calibrate / start / stop from the palm gesture (no controllers)."""
        self._gesture_active = False
        if not self.config.gesture.enabled:
            return
        hands = {s: frame.hands.get(s) if frame.controllers.get(s) is None else None for s in SIDES}
        wrists = {s: frame.hand_wrists.get(s) if hands[s] is not None else None for s in SIDES}
        self._gesture_active, event = self.gesture.update(hands, wrists, frame.headset, frame.now)
        if event == "short" and self.enabled:
            pass  # no short-hold stop while running; a long hold recalibrates
        elif event == "short" and self.calibrated:
            self.set_enabled(True)
            out.events.append("enabled")
        elif event is not None:
            # Home now; take the reference once the hands settle at home.
            self.set_enabled(False)
            for arm in self.arms.values():
                arm.reset()
            self.torso.reset()
            self._awaiting_reference = True
            out.events.append("calibrate")
        elif self._awaiting_reference and self.gesture.settled(wrists, self._gesture_active, frame.now):
            self.calibrate(frame)
            out.events.append("reference")

    def _step_arm(self, side: str, frame: InputFrame, out: OutputFrame, ctx: MappingContext) -> None:
        arm, clutch = self.arms[side], self.clutches[side]
        joy = frame.joys.get(side)
        controller = frame.controllers.get(side)
        hand_tracked = frame.hands.get(side) is not None
        source = controller
        if source is None and hand_tracked:
            source = frame.hand_wrists.get(side)

        if controller is None and (hand_tracked or self._lost_since[side] is not None
                                   or (arm.engaged and self.hand_mode.get(side) == "tracking")):
            self._step_hand_arm(side, frame, out, ctx)
            return
        grip = 0.0
        if joy is not None:
            grip = max(_axis(joy.axes, AXIS_GRIP), 1.0 if _button(joy.buttons, BUTTON_GRIP) else 0.0)
        want = clutch.update(grip, frame.now)

        if not self.enabled or not want or source is None:
            if arm.engaged:
                arm.release()
            if source is None:
                clutch.latched_off = False
            return
        if not arm.engaged:
            ee = frame.ee_poses.get(side)
            if ee is None:
                return
            arm.engage(source, ee, ctx, frame.now)
        out.arm_targets[side] = arm.update(source, frame.now, ctx)

    def _step_hand_arm(self, side: str, frame: InputFrame, out: OutputFrame,
                       ctx: MappingContext) -> None:
        """Hand tracking has no grip: engage once the mapped wrist meets the EE."""
        arm, cfg = self.arms[side], self.config
        wrist = frame.hand_wrists.get(side) if frame.hands.get(side) is not None else None
        if not self.enabled:
            arm.release()
            self._lost_since[side] = None
            return
        if wrist is None:
            if arm.engaged:
                lost = self._lost_since[side]
                if lost is None:
                    lost = self._lost_since[side] = frame.now
                if frame.now - lost < cfg.gesture.lost_hold_s:
                    out.arm_targets[side] = arm.target
                    return
                arm.release()
            self._lost_since[side] = None
            return
        self._lost_since[side] = None
        if not arm.engaged:
            ee = frame.ee_poses.get(side)
            if ee is None:
                return
            if arm.has_reference:
                arm.mapping.begin(wrist, ctx)
                desired = arm.mapping.desired(wrist, ctx)
                if desired is not None:
                    desired[:3, 3] = np.clip(desired[:3, 3], cfg.arm.workspace_min, cfg.arm.workspace_max)
                if not aligned(desired, ee, cfg.gesture, cfg.arm.rotation_enabled):
                    return
            arm.engage(wrist, ee, ctx, frame.now)
        out.arm_targets[side] = arm.update(wrist, frame.now, ctx)

    def _step_torso(self, frame: InputFrame, axes, out: OutputFrame) -> None:
        cfg = self.config.torso
        if not cfg.enabled or not self.enabled or frame.torso_home is None:
            self.torso.release()
            return
        follow = any(a.engaged for a in self.arms.values()) and frame.headset is not None
        if follow and not self.torso.engaged and frame.torso_pose is not None:
            self.torso.engage(frame.headset, frame.torso_pose, frame.torso_home, self._context(frame))
        elif not follow and self.torso.engaged:
            self.torso.release()
        stick = _axis(axes.get(cfg.stick_side) or (), AXIS_STICK_Y)
        out.torso_target = self.torso.update(
            frame.now, frame.headset, frame.torso_pose, frame.torso_home, stick)

    def _step_hand(self, side: str, frame: InputFrame, out: OutputFrame) -> None:
        if not self.config.enable_hands or not self.enabled:
            return
        points = frame.hands.get(side)
        joy = frame.joys.get(side)
        if points is not None:
            self.hand_mode[side] = "tracking"
            q = self.hand_retargeters[side].tracking(points, frame.now)
            # Fingers follow only while that arm is engaged, so free hand
            # motion (gestures, aligning) does not close the Revo2.
            arm = self.arms.get(side)
            if q is not None and (arm is None or arm.engaged):
                out.hand_positions[side] = q
        elif joy is not None:
            self.hand_mode[side] = "trigger"
            out.hand_positions[side] = retarget_trigger(_axis(joy.axes, AXIS_TRIGGER), self.config.hand)
        else:
            self.hand_mode[side] = "none"
