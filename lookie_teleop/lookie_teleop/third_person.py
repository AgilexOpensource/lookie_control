"""Third-person view: the headset hangs on the operator's chest.

The operator watches the robot directly, so the headset only tracks the
trunk. Its orientation is the body orientation (no neck), which gives clean
torso yaw/pitch, and the arms are mapped relative to the body: turning or
leaning moves the controllers together with the chest and leaves the arm
targets fixed relative to the robot torso, which follows the chest instead.

The mount rotation R_m (device vs. an upright body facing forward) is taken
at calibration, so the device may hang at any angle. Calibrate standing
upright and facing the robot's forward direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import MappingContext, body_pose, jump, mount_rotation


@dataclass
class ThirdPersonConfig:
    # Trunk pivot (~hip) relative to the headset, in the body frame (x forward,
    # z up). Leaning rotates the chest about it, so only crouching/stepping
    # moves it: torso height/forward follow the pivot, pitch the chest.
    pivot_offset_m: tuple[float, float, float] = (-0.1, 0.0, -0.45)


def _mount(ctx: MappingContext, headset: np.ndarray) -> np.ndarray:
    return ctx.mount if ctx.mount is not None else mount_rotation(headset, ctx.operator_yaw)


class BodyArmMapping:
    """Controller pose in the operator body frame -> EE pose in the torso frame.

        Q = B^-1 C,  A = (p_A0 + s (p_Q - p_Q0),  R_Q R_Q0^T R_A0),  E = F A

    with B the chest body frame, F the commanded torso frame and A0 = F0^-1 E0.
    """

    def __init__(self, config) -> None:
        self.config = config  # ArmMappingConfig
        self.clear()

    def clear(self) -> None:
        self._mount: np.ndarray | None = None
        self._q0: np.ndarray | None = None
        self._a0: np.ndarray | None = None
        self._last: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._q0 is not None

    def _relative(self, controller: np.ndarray, headset: np.ndarray) -> np.ndarray:
        return np.linalg.inv(body_pose(headset, self._mount)) @ np.asarray(controller, dtype=float)

    def set_reference(self, controller: np.ndarray, ee: np.ndarray, ctx: MappingContext) -> None:
        if ctx.headset is None:
            return
        self._mount = _mount(ctx, ctx.headset)
        self._q0 = self._relative(controller, ctx.headset)
        frame = ctx.torso_frame if ctx.torso_frame is not None else np.eye(4)
        self._a0 = np.linalg.inv(frame) @ np.asarray(ee, dtype=float)

    def begin(self, controller: np.ndarray, ctx: MappingContext) -> None:
        self._last = None
        if ctx.headset is not None and self.has_reference:
            self._last = self._relative(controller, ctx.headset)[:3, 3].copy()

    def desired(self, controller: np.ndarray, ctx: MappingContext) -> np.ndarray | None:
        if ctx.headset is None or not self.has_reference:
            return None
        cfg = self.config
        q = self._relative(controller, ctx.headset)
        step = jump(self._last, q[:3, 3], cfg.jump_reject_m)
        if step is not None:
            self._q0 = self._q0.copy()
            self._q0[:3, 3] += step
        self._last = q[:3, 3].copy()
        delta = (q[:3, 3] - self._q0[:3, 3]) * cfg.position_scale
        norm = float(np.linalg.norm(delta))
        if norm > cfg.max_offset_m > 0.0:
            delta *= cfg.max_offset_m / norm
        a = np.eye(4)
        a[:3, 3] = self._a0[:3, 3] + delta
        if cfg.rotation_enabled:
            a[:3, :3] = q[:3, :3] @ self._q0[:3, :3].T @ self._a0[:3, :3]
        else:
            a[:3, :3] = self._a0[:3, :3]
        frame = ctx.torso_frame if ctx.torso_frame is not None else np.eye(4)
        return frame @ a


class ChestTorsoSource:
    """Torso deltas [dx, dz, pitch, yaw] from the chest-mounted headset."""

    def __init__(self, config: ThirdPersonConfig) -> None:
        self.config = config
        self.offset = np.asarray(config.pivot_offset_m, dtype=float)
        self.clear()

    def clear(self) -> None:
        self._mount: np.ndarray | None = None
        self._r0 = np.eye(3)
        self._pivot0 = np.zeros(3)
        self._last: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._mount is not None

    def _body(self, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = body_pose(np.asarray(head, dtype=float), self._mount)
        return b[:3, :3], b[:3, 3] + b[:3, :3] @ self.offset

    def set_reference(self, head: np.ndarray, ctx: MappingContext) -> None:
        self._mount = _mount(ctx, np.asarray(head, dtype=float))
        self._r0, self._pivot0 = self._body(head)

    def begin(self, head: np.ndarray) -> None:
        self._last = np.asarray(head, dtype=float)[:3, 3].copy()

    def deltas(self, head: np.ndarray, jump_reject_m: float) -> np.ndarray:
        head = np.asarray(head, dtype=float)
        step = jump(self._last, head[:3, 3], jump_reject_m)
        if step is not None:
            self._pivot0 = self._pivot0 + step
        self._last = head[:3, 3].copy()
        r, pivot = self._body(head)
        rel = self._r0.T @ r  # = Rz(yaw) Ry(pitch) Rx(roll)
        yaw = math.atan2(rel[1, 0], rel[0, 0])
        pitch = math.atan2(-rel[2, 0], math.hypot(rel[0, 0], rel[1, 0]))
        d = self._r0.T @ (pivot - self._pivot0)
        return np.array([float(d[0]), float(d[2]), pitch, yaw])
