"""Small rigid-transform helpers. Quaternions are MuJoCo order (w, x, y, z)."""

import numpy as np


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    trace = np.trace(r)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def make_transform(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(quat)
    t[:3, 3] = position
    return t


def split_transform(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return t[:3, 3].copy(), matrix_to_quat(t[:3, :3])


def yaw_of(r: np.ndarray, forward=(1.0, 0.0, 0.0)) -> float:
    """Heading about +Z of the local ``forward`` axis of rotation ``r`` (Z-up)."""
    direction = np.asarray(r, dtype=float) @ np.asarray(forward, dtype=float)
    if np.hypot(direction[0], direction[1]) < 1e-6:
        return 0.0
    return float(np.arctan2(direction[1], direction[0]))


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(min(1.0, dot))
    q = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)
