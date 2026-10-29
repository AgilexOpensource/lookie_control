"""Map XR inputs to Revo2 hand joints.

Revo2 joint order used throughout:
    [thumb_metacarpal, thumb_proximal, index_proximal, middle_proximal,
     ring_proximal, pinky_proximal]
Distal joints are mimic joints and are handled by the controller.

Two sources are supported:
* Hand tracking: 25 WebXR joint positions (xr_media ``.../hand`` PoseArray).
  Per-joint metrics (pose and scale invariant):
    - finger: sum of bend angles between consecutive bones (MCP+PIP+DIP)
    - thumb curl: thumb MCP+IP bend
    - thumb opposition: thumb tip to pinky metacarpal distance / palm length
  :class:`HandRetargeter` adds, on top of the stateless mapping:
    - One-Euro filtering of the metrics and glitch rejection
    - per-operator ranges that widen to the observed extremes
    - a pinch term: when the thumb tip meets the index or middle tip, thumb
      and finger blend to a Revo2 pose whose tips touch; a completed pinch
      holds until the tips separate past a wider release distance
* Controller: the analog trigger closes each joint over its own trigger range,
  so the thumb opposes before it wraps over the fingers.
"""

from dataclasses import dataclass, field
import math

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)  # metacarpal, proximal, distal, tip
FINGERS = {
    "index": (5, 6, 7, 8, 9),
    "middle": (10, 11, 12, 13, 14),
    "ring": (15, 16, 17, 18, 19),
    "pinky": (20, 21, 22, 23, 24),
}
PINCH_FINGERS = ("index", "middle", "ring", "pinky")
# Revo2 joint index driven by each finger
FINGER_JOINT = {"index": 2, "middle": 3, "ring": 4, "pinky": 5}
NUM_JOINTS = 25
NUM_DOF = 6


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def chain_bend(points: np.ndarray, indices: tuple[int, ...]) -> float:
    """Sum of angles between consecutive bones along ``indices``."""
    bones = [_unit(points[b] - points[a]) for a, b in zip(indices[:-1], indices[1:])]
    total = 0.0
    for first, second in zip(bones[:-1], bones[1:]):
        total += float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))
    return total


def normalise(value: float, low: float, high: float) -> float:
    """Map ``value`` from [low, high] to [0, 1] (either direction), clamped."""
    if abs(high - low) < 1e-9:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class HandRetargetConfig:
    open_positions: list[float] = field(default_factory=lambda: [0.0] * 6)
    closed_positions: list[float] = field(
        default_factory=lambda: [1.57, 1.03, 1.41, 1.41, 1.41, 1.41])
    # Metric ranges (open, closed). They are deliberately narrow so any
    # operator reaches full open/close; with adaptive_range they widen to the
    # operator's observed extremes, but never past *_limits.
    finger_curl_range: tuple[float, float] = (0.5, 2.6)       # rad, bend sum
    thumb_curl_range: tuple[float, float] = (0.3, 1.1)        # rad, bend sum
    thumb_opposition_range: tuple[float, float] = (1.0, 0.7)  # tip-to-pinky / palm
    adaptive_range: bool = True
    finger_curl_limits: tuple[float, float] = (0.0, 3.6)
    thumb_curl_limits: tuple[float, float] = (0.0, 2.0)
    thumb_opposition_limits: tuple[float, float] = (1.6, 0.3)
    # Pinch: thumb-to-finger tip distance / palm length where blending starts
    # and is complete. For each finger in pinch_fingers, pinch_positions holds
    # [thumb_metacarpal, thumb_proximal, finger] that puts the Revo2 thumb and
    # finger tips together (found with the URDF FK). The Revo2 thumb cannot
    # reach the ring (9 mm) or pinky (24 mm) tips.
    pinch_enabled: bool = True
    pinch_range: tuple[float, float] = (0.45, 0.15)
    pinch_fingers: tuple[str, ...] = ("index", "middle")
    pinch_positions: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "index": (1.17, 0.71, 0.65), "middle": (1.49, 0.63, 0.73)})
    # Hysteresis (stateful retargeter only): once a pinch is complete it holds
    # until the distance ratio exceeds this value; <= pinch_range[0] disables.
    pinch_release: float = 0.6
    # Above this finger curl (0..1) the pinch fades out, so a fist stays a fist.
    pinch_max_curl: float = 0.8
    # One-Euro filter on the metrics (Hz, -, Hz); min_cutoff <= 0 disables.
    filter_min_cutoff: float = 2.0
    filter_beta: float = 0.3
    filter_d_cutoff: float = 1.0
    # Glitch rejection: hold the last output for frames whose palm length
    # differs from the filtered one by more than this fraction.
    palm_jump_ratio: float = 0.3
    trigger_deadzone: float = 0.02
    # Trigger interval (after the deadzone) over which each joint closes.
    trigger_ranges: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.5), (0.4, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)])


def hand_metrics(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (6 metrics, palm length, thumb-to-finger tip distances / palm).

    The distances follow PINCH_FINGERS order.
    """
    points = np.asarray(points, dtype=float).reshape(NUM_JOINTS, 3)
    palm = max(float(np.linalg.norm(points[FINGERS["middle"][1]] - points[WRIST])), 1e-3)
    metrics = [
        float(np.linalg.norm(points[THUMB[3]] - points[FINGERS["pinky"][0]])) / palm,
        chain_bend(points, THUMB),
    ]
    metrics += [chain_bend(points, FINGERS[name]) for name in ("index", "middle", "ring", "pinky")]
    pinch = np.array([np.linalg.norm(points[THUMB[3]] - points[FINGERS[name][4]])
                      for name in PINCH_FINGERS]) / palm
    return np.array(metrics), palm, pinch


def _ranges(config: HandRetargetConfig) -> list[tuple[float, float]]:
    return [tuple(config.thumb_opposition_range), tuple(config.thumb_curl_range)] + \
        [tuple(config.finger_curl_range)] * 4


def _limits(config: HandRetargetConfig) -> list[tuple[float, float]]:
    return [tuple(config.thumb_opposition_limits), tuple(config.thumb_curl_limits)] + \
        [tuple(config.finger_curl_limits)] * 4


def _map(metrics, pinch, ranges, config: HandRetargetConfig,
         engaged: set[str] | None = None) -> list[float]:
    """Map metrics to joints; ``engaged`` (updated in place) enables hysteresis."""
    alphas = [normalise(m, lo, hi) for m, (lo, hi) in zip(metrics, ranges)]
    q = _blend(alphas, config)
    if not config.pinch_enabled:
        if engaged is not None:
            engaged.clear()
        return q
    start, full = config.pinch_range
    release = max(float(config.pinch_release), start)
    weights = {}
    for name in config.pinch_fingers:
        d = float(pinch[PINCH_FINGERS.index(name)])
        if engaged is not None:
            # Engage once the blend is complete (w = 1 on both curves) and
            # release past the wider threshold (w = 0), so w stays continuous.
            if d <= full:
                engaged.add(name)
            elif d >= release:
                engaged.discard(name)
        lo = release if engaged is not None and name in engaged else start
        w = _smoothstep(normalise(d, lo, full))
        w *= 1.0 - _smoothstep(normalise(alphas[FINGER_JOINT[name]], config.pinch_max_curl, 1.0))
        weights[name] = w
    total = sum(weights.values())
    if total <= 1e-9:
        return q
    # Thumb goes to the weight-averaged target of the pinching fingers, each
    # finger to its own target.
    strength = max(weights.values())
    for j in (0, 1):
        target = sum(w * float(config.pinch_positions[n][j]) for n, w in weights.items()) / total
        q[j] += strength * (target - q[j])
    for name, w in weights.items():
        j = FINGER_JOINT[name]
        q[j] += w * (float(config.pinch_positions[name][2]) - q[j])
    return q


def retarget_hand_tracking(points: np.ndarray, config: HandRetargetConfig) -> list[float]:
    """Stateless mapping with the configured ranges (no filter, no adaptation,
    no pinch hysteresis)."""
    metrics, _, pinch = hand_metrics(points)
    return _map(metrics, pinch, _ranges(config), config)


def retarget_trigger(value: float, config: HandRetargetConfig) -> list[float]:
    """Return Revo2 joint positions from an analog trigger in [0, 1]."""
    value = float(np.clip(value, 0.0, 1.0))
    deadzone = float(np.clip(config.trigger_deadzone, 0.0, 0.49))
    base = normalise(value, deadzone, 1.0 - deadzone)
    ranges = list(config.trigger_ranges) or [(0.0, 1.0)] * len(config.open_positions)
    return _blend([normalise(base, lo, hi) for lo, hi in ranges], config)


def _blend(alphas: list[float], config: HandRetargetConfig) -> list[float]:
    return [
        float(o + a * (c - o))
        for o, c, a in zip(config.open_positions, config.closed_positions, alphas)
    ]


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) on a vector."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None or self.min_cutoff <= 0.0 or t <= self._t:
            if self._x is None or self.min_cutoff <= 0.0:
                self._x, self._dx = x.copy(), np.zeros_like(x)
            self._t = t if self._t is None or t > self._t else self._t
            return self._x.copy()
        dt = min(t - self._t, 0.5)
        self._t = t
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


class HandRetargeter:
    """Stateful hand-tracking retargeting for one hand."""

    MAX_REJECTED = 5

    def __init__(self, config: HandRetargetConfig) -> None:
        self.config = config
        c = config
        self._filter = OneEuro(c.filter_min_cutoff, c.filter_beta, c.filter_d_cutoff)
        self._palm = OneEuro(c.filter_min_cutoff, 0.0, c.filter_d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Forget the filter state and the learned ranges."""
        self._filter.reset()
        self._palm.reset()
        self._palm_value: float | None = None
        self._rejected = 0
        self.ranges = [list(r) for r in _ranges(self.config)]
        self.pinching: set[str] = set()
        self.output: list[float] | None = None

    def _widen(self, metrics: np.ndarray) -> None:
        for r, lims, m in zip(self.ranges, _limits(self.config), metrics):
            m = float(np.clip(m, min(lims), max(lims)))
            if r[1] >= r[0]:
                r[0], r[1] = min(r[0], m), max(r[1], m)
            else:
                r[0], r[1] = max(r[0], m), min(r[1], m)

    def tracking(self, points: np.ndarray, now: float) -> list[float] | None:
        """Return joint positions, or the last output for a rejected frame."""
        c = self.config
        metrics, palm, pinch = hand_metrics(points)
        if self._palm_value is not None and c.palm_jump_ratio > 0.0 and \
                abs(palm - self._palm_value) > c.palm_jump_ratio * self._palm_value:
            self._rejected += 1
            if self._rejected <= self.MAX_REJECTED:
                return self.output
            self._palm.reset()  # persistent change: accept the new hand size
        self._rejected = 0
        self._palm_value = float(self._palm(np.array([palm]), now)[0])
        values = self._filter(np.append(metrics, pinch), now)
        metrics, pinch = values[:NUM_DOF], values[NUM_DOF:]
        if c.adaptive_range:
            self._widen(metrics)
        self.output = _map(metrics, pinch, self.ranges, c, self.pinching)
        return self.output
