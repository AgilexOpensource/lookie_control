import numpy as np
import pytest

from lookie_teleop.hand_retarget import (
    FINGERS, HandRetargetConfig, HandRetargeter, THUMB, chain_bend, retarget_hand_tracking,
    retarget_trigger)


def open_hand() -> np.ndarray:
    """Flat hand: fingers along +X, spread along Y, thumb pointing out to +Y."""
    p = np.zeros((25, 3))
    for k, name in enumerate(("index", "middle", "ring", "pinky")):
        y = 0.03 - 0.02 * k
        for j, idx in enumerate(FINGERS[name]):
            p[idx] = [0.02 + 0.03 * j, y, 0.0]
    for j, idx in enumerate(THUMB):
        p[idx] = [0.02 + 0.01 * j, 0.05 + 0.03 * j, 0.0]
    return p


def fist() -> np.ndarray:
    """Fingers curled about the Y axis; thumb tip folded onto the pinky base."""
    p = open_hand()
    for name in ("index", "middle", "ring", "pinky"):
        idx = FINGERS[name]
        base = p[idx[1]].copy()
        angle = 0.0
        point = base
        for j in idx[2:]:
            angle += 1.2
            point = point + 0.03 * np.array([np.cos(angle), 0.0, -np.sin(angle)])
            p[j] = point
    p[THUMB[1]] = [0.05, 0.05, -0.01]
    p[THUMB[2]] = [0.08, 0.02, -0.03]
    p[THUMB[3]] = [0.05, -0.03, -0.03]
    return p


def test_chain_bend_straight_is_zero():
    assert chain_bend(open_hand(), FINGERS["index"]) == pytest.approx(0.0, abs=1e-9)


def test_open_and_fist():
    cfg = HandRetargetConfig()
    opened = retarget_hand_tracking(open_hand(), cfg)
    closed = retarget_hand_tracking(fist(), cfg)
    assert opened[2:] == pytest.approx([0.0] * 4)
    for i in range(2, 6):
        assert closed[i] > 1.2
    assert closed[0] > opened[0] + 0.5
    assert closed[1] > opened[1]
    for value, high in zip(closed, cfg.closed_positions):
        assert 0.0 <= value <= high + 1e-9


def test_scale_invariance():
    cfg = HandRetargetConfig()
    np.testing.assert_allclose(
        retarget_hand_tracking(fist(), cfg), retarget_hand_tracking(fist() * 1.3 + 0.5, cfg))


def test_trigger():
    cfg = HandRetargetConfig()
    assert retarget_trigger(0.0, cfg) == pytest.approx([0.0] * 6)
    assert retarget_trigger(1.0, cfg) == pytest.approx(cfg.closed_positions)
    half = retarget_trigger(0.5, cfg)
    assert half[2] == pytest.approx(1.41 * (0.5 - 0.1) / 0.9)
    # The thumb opposes first and is fully rotated before the fingers close
    assert retarget_trigger(0.25, cfg)[0] / 1.57 > retarget_trigger(0.25, cfg)[2] / 1.41
    assert retarget_trigger(0.51, cfg)[0] == pytest.approx(1.57)
    assert retarget_trigger(0.3, cfg)[1] == pytest.approx(0.0)


def pinch_hand(finger: str = "index", gap: float = 0.005) -> np.ndarray:
    """``finger`` slightly curled, thumb tip ``gap`` metres from its tip."""
    p = open_hand()
    idx = FINGERS[finger]
    point, angle = p[idx[1]].copy(), 0.0
    for j in idx[2:]:
        angle += 0.35
        point = point + 0.03 * np.array([np.cos(angle), 0.0, -np.sin(angle)])
        p[j] = point
    tip = p[idx[4]]
    p[THUMB[2]] = [0.06, 0.05, -0.02]
    p[THUMB[3]] = tip + [0.0, 0.0, -gap]
    return p


def _pose(cfg: HandRetargetConfig, finger: str) -> list[float]:
    return list(cfg.pinch_positions[finger])


def test_pinch_blends_to_touching_pose():
    cfg = HandRetargetConfig()
    q = retarget_hand_tracking(pinch_hand(), cfg)
    assert q[:3] == pytest.approx(_pose(cfg, "index"), abs=1e-6)
    off = retarget_hand_tracking(pinch_hand(), HandRetargetConfig(pinch_enabled=False))
    assert off[:3] != pytest.approx(_pose(cfg, "index"), abs=0.05)
    # A fist also brings the thumb near the index but must stay closed
    fist_q = retarget_hand_tracking(fist(), cfg)
    assert fist_q[2] > 1.2


def test_middle_finger_pinch():
    cfg = HandRetargetConfig()
    q = retarget_hand_tracking(pinch_hand("middle"), cfg)
    target = _pose(cfg, "middle")
    assert q[:2] == pytest.approx(target[:2], abs=1e-6)
    assert q[3] == pytest.approx(target[2], abs=1e-6)
    # The index is not pinching and keeps its own (straight) mapping
    assert q[2] == pytest.approx(0.0, abs=1e-6)
    only_index = retarget_hand_tracking(pinch_hand("middle"), HandRetargetConfig(pinch_fingers=("index",)))
    assert only_index[3] != pytest.approx(target[2], abs=0.05)


def test_pinch_hysteresis():
    cfg = HandRetargetConfig(filter_min_cutoff=0.0, adaptive_range=False, palm_jump_ratio=0.0)
    palm = float(np.linalg.norm(open_hand()[FINGERS["middle"][1]]))
    between = pinch_hand(gap=0.5 * palm)  # ratio 0.5: past pinch_range start, below release
    free = retarget_hand_tracking(between, HandRetargetConfig(pinch_enabled=False))
    assert retarget_hand_tracking(between, cfg) == pytest.approx(free, abs=1e-6)

    r = HandRetargeter(cfg)
    assert r.tracking(between, 0.0) == pytest.approx(free, abs=1e-6)
    assert r.tracking(pinch_hand(), 0.1)[:3] == pytest.approx(_pose(cfg, "index"), abs=1e-6)
    assert r.pinching == {"index"}
    # Opening to 0.5 keeps a partial pinch instead of dropping it
    held = r.tracking(between, 0.2)
    assert free[0] + 0.05 < held[0] < _pose(cfg, "index")[0]
    # Past the release distance the pinch is gone
    r.tracking(pinch_hand(gap=0.7 * palm), 0.3)
    assert r.pinching == set()
    assert r.tracking(between, 0.4) == pytest.approx(free, abs=1e-6)


def test_pinch_weight_is_continuous():
    cfg = HandRetargetConfig(filter_min_cutoff=0.0, adaptive_range=False, palm_jump_ratio=0.0)
    palm = float(np.linalg.norm(open_hand()[FINGERS["middle"][1]]))
    r = HandRetargeter(cfg)
    gaps = np.concatenate([np.linspace(0.8, 0.05, 80), np.linspace(0.05, 0.8, 80)])
    qs = np.array([r.tracking(pinch_hand(gap=g * palm), 0.01 * i) for i, g in enumerate(gaps)])
    assert np.max(np.abs(np.diff(qs, axis=0))) < 0.1


def test_retargeter_filters_and_rejects_glitches():
    cfg = HandRetargetConfig(adaptive_range=False)
    r = HandRetargeter(cfg)
    for i in range(20):
        r.tracking(open_hand(), 0.02 * i)
    first = r.tracking(fist(), 0.4)
    assert 0.0 < first[2] < 1.0  # filtered, not a jump to closed
    for i in range(1, 60):
        q = r.tracking(fist(), 0.4 + 0.02 * i)
    assert q == pytest.approx(retarget_hand_tracking(fist(), cfg), abs=1e-3)
    # A frame with a 2x palm (tracking glitch) is ignored
    glitch = open_hand()
    glitch[FINGERS["middle"][1]] *= 2.0
    assert r.tracking(glitch, 2.0) == q
    # ... unless it persists
    for i in range(1, 10):
        r.tracking(glitch * 1.0, 2.0 + 0.02 * i)
    assert r.tracking(glitch, 2.3) != q


def test_adaptive_range_widens_to_operator_extremes():
    cfg = HandRetargetConfig(filter_min_cutoff=0.0, pinch_enabled=False)
    r = HandRetargeter(cfg)
    # A stiff operator whose fist only reaches 2.2 rad bend: still closes fully
    base = cfg.finger_curl_range
    r.tracking(open_hand(), 0.0)
    assert r.ranges[2][0] == pytest.approx(0.0)  # flat hand widened the open end
    assert r.ranges[2][1] == pytest.approx(base[1])
    q = r.tracking(fist(), 0.1)
    assert q[2] == pytest.approx(1.41)
    r.reset()
    assert r.ranges[2] == pytest.approx(list(base))
