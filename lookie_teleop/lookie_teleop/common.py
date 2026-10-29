"""Pieces shared by the first- and third-person mappings."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transforms import rot_z


@dataclass
class MappingContext:
    """Per-tick inputs a mapping may use besides the controller pose."""

    operator_yaw: float = 0.0
    headset: np.ndarray | None = None
    # Commanded torso frame in the base frame (see :func:`torso_frame`)
    torso_frame: np.ndarray | None = None
    # Headset mount rotation R_m (body -> device) recorded at calibration
    mount: np.ndarray | None = None


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def torso_frame(offsets: np.ndarray, home: np.ndarray | None) -> np.ndarray:
    """Torso frame for offsets [x, z, pitch, yaw] from home: Rz(yaw) Ry(pitch).

    The home orientation is left out so this frame stays upright at home; arm
    poses expressed in it are composed back with the same frame.
    """
    out = np.eye(4)
    if home is not None:
        out[:3, 3] = home[:3, 3]
    out[:3, 3] += np.array([offsets[0], 0.0, offsets[1]])
    out[:3, :3] = rot_z(offsets[3]) @ rot_y(offsets[2])
    return out


def mount_rotation(headset: np.ndarray, operator_yaw: float) -> np.ndarray:
    """Device orientation relative to an upright body facing ``operator_yaw``."""
    return rot_z(operator_yaw).T @ headset[:3, :3]


def body_pose(headset: np.ndarray, mount: np.ndarray) -> np.ndarray:
    """Operator body frame (x forward, z up) at the headset position."""
    out = np.eye(4)
    out[:3, 3] = headset[:3, 3]
    out[:3, :3] = headset[:3, :3] @ mount.T
    return out


def jump(last: np.ndarray | None, current: np.ndarray, threshold: float) -> np.ndarray | None:
    """Position step larger than ``threshold`` (a tracking glitch), else None."""
    if last is None or threshold <= 0.0:
        return None
    step = current - last
    return step if float(np.linalg.norm(step)) > threshold else None
