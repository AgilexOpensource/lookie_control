"""First-person view: the headset is worn on the head.

Arms map controller motion in the world, rotated by the operator heading at
calibration. The torso follows the head; because head motion includes the
neck, trunk lean is estimated from forward head travel by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import MappingContext, jump
from .transforms import rot_z, yaw_of


@dataclass
class FirstPersonConfig:
    # "translation": trunk lean from forward head motion,
    # pitch = atan(dx / lean_radius_m); ignores head nods (neck).
    # "head": headset forward-axis elevation (includes neck nods).
    pitch_source: str = "translation"
    lean_radius_m: float = 0.6


class WorldArmMapping:
    """p = p_E0 + s R_yaw^T (p_C - p_C0),  R = (R_yaw^T R_C R_C0^T R_yaw) R_E0."""

    def __init__(self, config) -> None:
        self.config = config  # ArmMappingConfig
        self.clear()

    def clear(self) -> None:
        self._c0: np.ndarray | None = None
        self._e0: np.ndarray | None = None
        self._yaw = np.eye(3)
        self._last: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._c0 is not None

    def set_reference(self, controller: np.ndarray, ee: np.ndarray, ctx: MappingContext) -> None:
        self._c0 = np.asarray(controller, dtype=float).copy()
        self._e0 = np.asarray(ee, dtype=float).copy()
        self._yaw = rot_z(ctx.operator_yaw)

    def begin(self, controller: np.ndarray, ctx: MappingContext) -> None:
        self._last = np.asarray(controller, dtype=float)[:3, 3].copy()

    def desired(self, controller: np.ndarray, ctx: MappingContext) -> np.ndarray | None:
        cfg = self.config
        controller = np.asarray(controller, dtype=float)
        p = controller[:3, 3]
        step = jump(self._last, p, cfg.jump_reject_m)
        if step is not None:
            self._c0 = self._c0.copy()
            self._c0[:3, 3] += step
        self._last = p.copy()
        delta = self._yaw.T @ (p - self._c0[:3, 3]) * cfg.position_scale
        norm = float(np.linalg.norm(delta))
        if norm > cfg.max_offset_m > 0.0:
            delta *= cfg.max_offset_m / norm
        out = np.eye(4)
        out[:3, 3] = self._e0[:3, 3] + delta
        if cfg.rotation_enabled:
            rel = controller[:3, :3] @ self._c0[:3, :3].T
            out[:3, :3] = self._yaw.T @ rel @ self._yaw @ self._e0[:3, :3]
        else:
            out[:3, :3] = self._e0[:3, :3]
        return out


class HeadTorsoSource:
    """Torso deltas [dx, dz, pitch, yaw] from the head-worn headset."""

    def __init__(self, config: FirstPersonConfig, forward_axis) -> None:
        if config.pitch_source not in ("translation", "head"):
            raise ValueError(f"unknown first_person.pitch_source '{config.pitch_source}'")
        self.config = config
        self.forward_axis = np.asarray(forward_axis, dtype=float)
        self.clear()

    def clear(self) -> None:
        self._head0: np.ndarray | None = None
        self._pitch0 = 0.0
        self._yaw0 = 0.0
        self._frame_yaw = 0.0
        self._last: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._head0 is not None

    def _pitch(self, head: np.ndarray) -> float:
        f = head[:3, :3] @ self.forward_axis
        return math.atan2(-f[2], math.hypot(f[0], f[1]))

    def set_reference(self, head: np.ndarray, ctx: MappingContext) -> None:
        self._head0 = np.asarray(head, dtype=float).copy()
        self._pitch0 = self._pitch(self._head0)
        self._yaw0 = yaw_of(self._head0[:3, :3], self.forward_axis)
        self._frame_yaw = ctx.operator_yaw

    def begin(self, head: np.ndarray) -> None:
        self._last = np.asarray(head, dtype=float)[:3, 3].copy()

    def deltas(self, head: np.ndarray, jump_reject_m: float) -> np.ndarray:
        head = np.asarray(head, dtype=float)
        step = jump(self._last, head[:3, 3], jump_reject_m)
        if step is not None:
            self._head0 = self._head0.copy()
            self._head0[:3, 3] += step
        self._last = head[:3, 3].copy()
        d = rot_z(self._frame_yaw).T @ (head[:3, 3] - self._head0[:3, 3])
        if self.config.pitch_source == "translation":
            pitch = math.atan2(float(d[0]), self.config.lean_radius_m)
        else:
            pitch = self._pitch(head) - self._pitch0
        yaw = math.remainder(yaw_of(head[:3, :3], self.forward_axis) - self._yaw0, math.tau)
        return np.array([float(d[0]), float(d[2]), pitch, yaw])
