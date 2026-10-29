"""Hand-tracking teleop control without controllers.

Hand tracking has no buttons and the fingers are retargeted to the Revo2, so
calibration, start and stop use a whole-hand gesture that does not occur while
manipulating: both palms open and facing the headset, held still.

    short (hold_s .. long_s, fires on release):
        not calibrated -> calibrate;  calibrated -> start;  running -> ignored
    long (>= long_s, fires while held): calibrate

Calibrate homes the robot; the reference is taken once both wrists settle in
the home posture afterwards.

An arm joins only once the operator's wrist, mapped through the calibration,
matches the robot EE (:func:`aligned`), so there is no jump on engage. This
works with either view mode: the headset may be on the head or on the chest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hand_retarget import FINGERS, WRIST, chain_bend


@dataclass
class GestureConfig:
    enabled: bool = True
    hold_s: float = 1.0
    long_s: float = 3.0
    # Palm normal vs. direction to the headset: cos of the max angle.
    palm_facing_cos: float = math.cos(math.radians(45.0))
    # Max mean finger bend (rad, MCP+PIP+DIP) for an "open" hand.
    open_curl_max: float = 1.0
    # Max wrist speed (m/s) while the gesture is held.
    still_speed: float = 0.15
    # A wrist moving farther than this from where the hold began ends it at
    # once, so a palm turned to the headset mid-motion never counts.
    still_drift_m: float = 0.06
    # Likewise for the fingers: any finger bending more than this (rad,
    # MCP+PIP+DIP) from the hold start ends the hold.
    still_finger_rad: float = 0.35
    # Tracking flicker shorter than this does not end the hold.
    release_s: float = 0.3
    # After a gesture, it must be released this long before the next one.
    rearm_s: float = 0.5
    # Calibration reference: wrists below settle_speed for settle_s.
    settle_s: float = 0.8
    settle_speed: float = 0.05
    # Engage once the mapped wrist is within these of the measured EE.
    align_position_m: float = 0.05
    align_angle_rad: float = math.radians(15.0)
    # Tracking dropouts shorter than this hold the arm; longer release it.
    lost_hold_s: float = 0.2


def palm_normal(points: np.ndarray, side: str) -> np.ndarray:
    """Unit normal out of the palm (WebXR joints, right-handed world)."""
    points = np.asarray(points, dtype=float)
    a = points[FINGERS["index"][0]] - points[WRIST]
    b = points[FINGERS["pinky"][0]] - points[WRIST]
    n = np.cross(a, b)
    if side == "left":
        n = -n
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 1e-9 else n


def finger_bends(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.array([chain_bend(points, FINGERS[f][1:]) for f in FINGERS])


def palm_facing(points: np.ndarray, side: str, headset: np.ndarray, config: GestureConfig) -> bool:
    """Open hand with the palm towards the headset."""
    points = np.asarray(points, dtype=float)
    curl = float(np.mean(finger_bends(points)))
    if curl > config.open_curl_max:
        return False
    centre = points[[WRIST, FINGERS["index"][0], FINGERS["pinky"][0]]].mean(axis=0)
    to_head = np.asarray(headset, dtype=float)[:3, 3] - centre
    dist = float(np.linalg.norm(to_head))
    if dist < 1e-6:
        return False
    return float(np.dot(palm_normal(points, side), to_head / dist)) >= config.palm_facing_cos


def aligned(desired: np.ndarray | None, ee: np.ndarray | None, config: GestureConfig,
            rotation: bool = True) -> bool:
    if desired is None or ee is None:
        return False
    if float(np.linalg.norm(desired[:3, 3] - ee[:3, 3])) > config.align_position_m:
        return False
    if not rotation:
        return True
    cos = (float(np.trace(desired[:3, :3].T @ ee[:3, :3])) - 1.0) / 2.0
    return math.acos(float(np.clip(cos, -1.0, 1.0))) <= config.align_angle_rad


class _Speed:
    """Wrist speed from consecutive samples."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last: np.ndarray | None = None
        self._t: float | None = None
        self.value = math.inf

    def update(self, p: np.ndarray, now: float) -> float:
        p = np.asarray(p, dtype=float)
        if self._last is not None and self._t is not None and now > self._t:
            self.value = float(np.linalg.norm(p - self._last)) / (now - self._t)
        self._last, self._t = p.copy(), now
        return self.value


class GestureControl:
    """Both-palms gesture timer and calibration settle detector."""

    def __init__(self, config: GestureConfig) -> None:
        self.config = config
        self._speed = {s: _Speed() for s in ("left", "right")}
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self._last_active: float | None = None
        self._anchor: dict = {}
        self._finger_anchor: dict = {}
        self._armed = True
        self._released_at: float | None = None
        self._still_since: float | None = None
        for s in self._speed.values():
            s.reset()

    def update(self, hands: dict, wrists: dict, headset: np.ndarray | None,
               now: float) -> tuple[bool, str | None]:
        """Return (gesture held now, None / "short" / "long")."""
        cfg = self.config
        speeds = {}
        for side, speed in self._speed.items():
            w = wrists.get(side)
            if w is None:
                speed.reset()
            else:
                speeds[side] = speed.update(np.asarray(w)[:3, 3], now)

        active = (
            headset is not None
            and all(hands.get(s) is not None for s in self._speed)
            and all(palm_facing(hands[s], s, headset, cfg) for s in self._speed)
            and all(speeds.get(s, math.inf) <= cfg.still_speed for s in self._speed))
        moved = self._since is not None and any(
            s in self._anchor and wrists.get(s) is not None
            and float(np.linalg.norm(np.asarray(wrists[s])[:3, 3] - self._anchor[s])) > cfg.still_drift_m
            for s in self._speed)
        moved = moved or (self._since is not None and any(
            s in self._finger_anchor and hands.get(s) is not None
            and float(np.max(np.abs(finger_bends(hands[s]) - self._finger_anchor[s]))) > cfg.still_finger_rad
            for s in self._speed))
        if moved:
            active = False
        if not active:
            if (not moved and self._since is not None and self._last_active is not None
                    and now - self._last_active < cfg.release_s):
                return True, None
            event = None
            if self._since is not None and self._last_active - self._since >= cfg.hold_s:
                event = "short"
                self._armed = False
            self._since = None
            self._anchor = {}
            self._finger_anchor = {}
            if not self._armed:
                if self._released_at is None:
                    self._released_at = now
                if now - self._released_at >= cfg.rearm_s:
                    self._armed = True
                    self._released_at = None
            return False, event
        self._released_at = None
        self._last_active = now
        if not self._armed:
            return True, None
        if self._since is None:
            self._since = now
            self._anchor = {s: np.asarray(wrists[s])[:3, 3].copy() for s in self._speed}
            self._finger_anchor = {s: finger_bends(hands[s]) for s in self._speed}
        if now - self._since >= cfg.long_s:
            self._since = None
            self._anchor = {}
            self._finger_anchor = {}
            self._armed = False
            return True, "long"
        return True, None

    @property
    def stage(self) -> str | None:
        """None, "hold" (too short yet) or "ready" (release now for short)."""
        if self._since is None or self._last_active is None:
            return None
        return "ready" if self._last_active - self._since >= self.config.hold_s else "hold"

    def settled(self, wrists: dict, gesture_active: bool, now: float) -> bool:
        """Both wrists tracked and still for settle_s, gesture released."""
        cfg = self.config
        still = (not gesture_active and self._armed
                 and all(wrists.get(s) is not None for s in self._speed)
                 and all(self._speed[s].value <= cfg.settle_speed for s in self._speed))
        if not still:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = now
        if now - self._still_since >= cfg.settle_s:
            self._still_since = None
            return True
        return False
