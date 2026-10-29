"""Build a kinematic MuJoCo model from a Lookie URDF.

The URDF visual meshes are Collada (``.dae``), which MuJoCo cannot load, so the
collision STL meshes are used as geometry. STL files above MuJoCo's face limit
are replaced by their bounding box. A planar base (x, y, yaw) and mocap target
markers are added with ``MjSpec``. Contacts are disabled; the model is only
used for forward kinematics, Jacobians and visualization.
"""

from dataclasses import dataclass, field
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

MAX_STL_FACES = 200000
BASE_JOINTS = ("base_x", "base_y", "base_yaw")


@dataclass(frozen=True)
class Mimic:
    joint: str
    source: str
    multiplier: float
    offset: float


@dataclass
class RobotModel:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    root_body: str
    urdf_joints: list[str]
    mimics: list[Mimic] = field(default_factory=list)
    marker_bodies: dict[str, str] = field(default_factory=dict)


def _resolve(filename: str, package_dirs: dict[str, str]) -> str:
    if filename.startswith("package://"):
        package, _, rest = filename[len("package://"):].partition("/")
        if package not in package_dirs:
            raise ValueError(f"unknown package '{package}' in mesh path {filename}")
        return os.path.join(package_dirs[package], rest)
    if filename.startswith("file://"):
        return filename[len("file://"):]
    return filename


def stl_bounds(path: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Return face count and vertex bounds of a binary STL file."""
    faces = int(np.fromfile(path, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    vertices = np.fromfile(path, dtype=record, count=faces, offset=84)["v"]
    vertices = vertices.reshape(-1, 3)
    return faces, vertices.min(axis=0), vertices.max(axis=0)


def _prepare_urdf(urdf_path: str, package_dirs: dict[str, str]) -> tuple[str, list[str], list[Mimic], str]:
    root = ET.parse(urdf_path).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [l.get("name") for l in root.findall("link") if l.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")

    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                continue
            path = _resolve(mesh.get("filename"), package_dirs)
            faces, low, high = stl_bounds(path)
            if faces <= MAX_STL_FACES:
                mesh.set("filename", path)
                continue
            # Replace oversized meshes with their bounding box.
            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            low, high = low * scale, high * scale
            geometry.remove(mesh)
            ET.SubElement(geometry, "box", size=" ".join(f"{v:.5f}" for v in high - low))
            origin = collision.find("origin")
            if origin is None:
                origin = ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            if np.allclose([float(v) for v in origin.get("rpy", "0 0 0").split()], 0.0):
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                origin.set("xyz", " ".join(f"{v:.5f}" for v in xyz + (low + high) / 2))

    joints, mimics = [], []
    for joint in root.findall("joint"):
        if joint.get("type") in ("fixed", "floating", "planar"):
            continue
        joints.append(joint.get("name"))
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics.append(Mimic(
                joint.get("name"), mimic.get("joint"),
                float(mimic.get("multiplier", 1.0)), float(mimic.get("offset", 0.0)),
            ))

    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension, "compiler",
        fusestatic="false", discardvisual="false", strippath="false", balanceinertia="true",
    )
    return ET.tostring(root, encoding="unicode"), joints, mimics, roots[0]


def build_robot_model(
    urdf_path: str,
    package_dirs: dict[str, str],
    markers: dict[str, tuple[float, float, float, float]] | None = None,
) -> RobotModel:
    """Compile the URDF into a MuJoCo model with a planar base and markers.

    ``markers`` maps a marker name to an RGBA colour; each becomes a mocap body
    named ``marker_<name>`` with a sphere and RGB axis cylinders.
    """
    xml, joints, mimics, root_name = _prepare_urdf(urdf_path, package_dirs)
    spec = mujoco.MjSpec.from_string(xml)
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)

    root = spec.body(root_name)
    for name, kind, axis in (
        ("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
        ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
        ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1]),
    ):
        root.add_joint(name=name, type=kind, axis=axis)
    # Robot geoms stay collidable at compile time so MuJoCo builds convex hulls
    # (needed for fast IK self-collision distances); contacts are disabled
    # globally and contype/conaffinity are cleared on the compiled model.
    robot_geoms = list(spec.geoms)
    for geom in robot_geoms:
        geom.contype = 1
        geom.conaffinity = 1

    world = spec.worldbody
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.85, 0.87, 0.9, 1],
        contype=0, conaffinity=0,
    )
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    marker_bodies = {}
    for name, rgba in (markers or {}).items():
        body = world.add_body(name=f"marker_{name}", mocap=True)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=list(rgba),
            contype=0, conaffinity=0, group=2,
        )
        for axis_rgba, quat in (
            ([1, 0, 0, 0.8], [0.7071068, 0, 0.7071068, 0]),
            ([0, 1, 0, 0.8], [0.7071068, -0.7071068, 0, 0]),
            ([0, 0, 1, 0.8], [1, 0, 0, 0]),
        ):
            # Cylinder along its local z, shifted to start at the marker origin.
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.array(quat, dtype=float))
            offset = rot.reshape(3, 3)[:, 2] * 0.05
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.05, 0],
                pos=offset.tolist(), quat=quat, rgba=axis_rgba,
                contype=0, conaffinity=0, group=2,
            )
        marker_bodies[name] = body.name

    model = spec.compile()
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return RobotModel(spec, model, root_name, joints, mimics, marker_bodies)
