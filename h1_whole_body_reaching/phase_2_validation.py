# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent H1-2 URDF kinematics and static-asset validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    """A URDF revolute joint connecting two links."""

    name: str
    parent: str
    child: str
    origin_xyz_m: np.ndarray
    origin_rpy_rad: np.ndarray
    axis: np.ndarray


def _vector(text: str | None, count: int) -> np.ndarray:
    """Convert a space-separated XML vector to a NumPy vector."""
    values = np.fromstring(text or "", sep=" ")
    if values.shape != (count,):
        raise ValueError(f"Expected {count} values, received {text!r}.")
    return values


def _rotation_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = rpy_rad
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), (-sp, cp * sr, cp * cr)))


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return Rodrigues' rotation about a unit joint axis."""
    axis = axis / np.linalg.norm(axis)
    skew = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def parse_urdf(urdf_path: Path) -> tuple[dict[str, JointSpec], dict[str, tuple[float, np.ndarray]], dict[str, int]]:
    """Parse joints, inertials, and collision counts from a URDF.

    Returns:
        Joint mapping, link mass/CoM mapping, and collision geometry count per link.
    """
    root = ElementTree.parse(urdf_path).getroot()
    joints: dict[str, JointSpec] = {}
    for joint in root.findall("joint"):
        if joint.get("type") in {"fixed", None}:
            continue
        origin = joint.find("origin")
        joints[joint.attrib["name"]] = JointSpec(
            name=joint.attrib["name"], parent=joint.find("parent").attrib["link"], child=joint.find("child").attrib["link"],
            origin_xyz_m=_vector(origin.get("xyz") if origin is not None else "0 0 0", 3),
            origin_rpy_rad=_vector(origin.get("rpy") if origin is not None else "0 0 0", 3),
            axis=_vector(joint.find("axis").get("xyz"), 3),
        )
    inertials: dict[str, tuple[float, np.ndarray]] = {}
    collisions: dict[str, int] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        collisions[name] = len(link.findall("collision"))
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").attrib["value"])
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        tensor = np.array(((float(inertia.attrib["ixx"]), float(inertia.attrib["ixy"]), float(inertia.attrib["ixz"])), (float(inertia.attrib["ixy"]), float(inertia.attrib["iyy"]), float(inertia.attrib["iyz"])), (float(inertia.attrib["ixz"]), float(inertia.attrib["iyz"]), float(inertia.attrib["izz"])) ))
        if mass <= 0.0 or np.min(np.linalg.eigvalsh(tensor)) <= 0.0:
            raise ValueError(f"Link {name} has non-positive mass or inertia.")
        inertials[name] = (mass, _vector(origin.get("xyz") if origin is not None else "0 0 0", 3))
    return joints, inertials, collisions


def _forward_kinematic_transforms(urdf_path: Path, joint_positions_rad: dict[str, float], root_position_w_m: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute independent URDF link rotations and origins in world coordinates."""
    joints, _, _ = parse_urdf(urdf_path)
    children: dict[str, list[JointSpec]] = {}
    child_links = set()
    for joint in joints.values():
        children.setdefault(joint.parent, []).append(joint)
        child_links.add(joint.child)
    roots = set(children) - child_links
    if roots != {"pelvis"}:
        raise ValueError(f"Expected pelvis root, got {sorted(roots)}.")
    transforms = {"pelvis": (np.eye(3), np.asarray(root_position_w_m, dtype=float))}
    pending = ["pelvis"]
    while pending:
        parent = pending.pop()
        parent_rotation, parent_position = transforms[parent]
        for joint in children.get(parent, []):
            rotation = parent_rotation @ _rotation_rpy(joint.origin_rpy_rad) @ _axis_rotation(joint.axis, joint_positions_rad.get(joint.name, 0.0))
            position = parent_position + parent_rotation @ joint.origin_xyz_m
            transforms[joint.child] = (rotation, position)
            pending.append(joint.child)
    return transforms


def forward_kinematics(urdf_path: Path, joint_positions_rad: dict[str, float], root_position_w_m: np.ndarray) -> dict[str, np.ndarray]:
    """Compute independent URDF link origins in world coordinates [m]."""
    return {name: transform[1] for name, transform in _forward_kinematic_transforms(urdf_path, joint_positions_rad, root_position_w_m).items()}


def static_com_and_support_margin(urdf_path: Path, joint_positions_rad: dict[str, float], root_position_w_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute URDF static CoM and its lateral margin to ankle-roll midpoint [m]."""
    joints, inertials, _ = parse_urdf(urdf_path)
    transforms = _forward_kinematic_transforms(urdf_path, joint_positions_rad, root_position_w_m)
    total_mass = sum(mass for mass, _ in inertials.values())
    com = sum((mass * (transforms[name][1] + transforms[name][0] @ local_com) for name, (mass, local_com) in inertials.items()), np.zeros(3)) / total_mass
    left_y = transforms["left_ankle_roll_link"][1][1]
    right_y = transforms["right_ankle_roll_link"][1][1]
    return com, min(left_y - com[1], com[1] - right_y)
