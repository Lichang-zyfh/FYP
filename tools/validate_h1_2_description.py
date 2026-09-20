#!/usr/bin/env python3
"""Validate the static interface of the official Unitree H1-2 description."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as element_tree


EXPECTED_JOINTS = (
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
WRIST_JOINTS = frozenset(name for name in EXPECTED_JOINTS if "wrist" in name)
REQUIRED_LINKS = frozenset({"pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "right_wrist_yaw_link"})


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(description_dir: Path) -> dict[str, object]:
    """Validate the H1-2 handless URDF and return a machine-readable report."""
    urdf_path = description_dir / "h1_2_handless.urdf"
    root = element_tree.parse(urdf_path).getroot()
    joints = [joint for joint in root.findall("joint") if joint.get("type") not in {None, "fixed"}]
    joint_names = tuple(joint.attrib["name"] for joint in joints)
    links = {link.attrib["name"] for link in root.findall("link")}
    child_links = {child.attrib["link"] for joint in root.findall("joint") for child in joint.findall("child")}
    parent_links = {parent.attrib["link"] for joint in root.findall("joint") for parent in joint.findall("parent")}
    root_links = sorted(parent_links - child_links)
    errors: list[str] = []
    if joint_names != EXPECTED_JOINTS:
        errors.append("Actuated joint names or ordering differs from the frozen official interface.")
    if len(joint_names) != 27:
        errors.append(f"Expected 27 actuated joints, found {len(joint_names)}.")
    if len([name for name in joint_names if name not in WRIST_JOINTS]) != 21:
        errors.append("Excluding both wrists did not leave 21 policy joints.")
    if root_links != ["pelvis"]:
        errors.append(f"Expected pelvis as the only root link, found {root_links}.")
    missing_links = sorted(REQUIRED_LINKS - links)
    if missing_links:
        errors.append(f"Required links missing: {missing_links}.")
    missing_limits = sorted(joint.attrib["name"] for joint in joints if joint.find("limit") is None)
    if missing_limits:
        errors.append(f"Actuated joints without limits: {missing_limits}.")
    return {
        "urdf": str(urdf_path),
        "sha256": sha256(urdf_path),
        "actuated_joint_count": len(joint_names),
        "policy_joint_count": len([name for name in joint_names if name not in WRIST_JOINTS]),
        "root_links": root_links,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }


def main() -> int:
    """Run the command-line validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description_dir", type=Path, help="Path containing h1_2_handless.urdf")
    args = parser.parse_args()
    report = validate(args.description_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
