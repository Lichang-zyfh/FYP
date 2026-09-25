#!/usr/bin/env python3
"""Validate H1-2 asset dynamics, joint directions, symmetry, FK, and static support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES
from h1_whole_body_reaching.phase_2_validation import forward_kinematics, parse_urdf, static_com_and_support_margin
from h1_whole_body_reaching.robot_cfg import FIXED_BASE_ROOT_POSITION_W_M


def validate(urdf_path: Path, mjcf_path: Path, perturbation_rad: float = 0.02) -> dict[str, object]:
    """Run independent asset checks and return a JSON-ready report."""
    joints, inertials, collisions = parse_urdf(urdf_path)
    if set(joints) != set(ALL_JOINT_NAMES):
        raise RuntimeError("URDF joints do not match the frozen H1-2 interface.")
    support_links = ("left_ankle_roll_link", "right_ankle_roll_link")
    if any(collisions[link] == 0 for link in support_links):
        raise RuntimeError("A candidate support link has no collision geometry.")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    kinematic_root_position = data.xpos[pelvis_id].copy()
    zero_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
    fk = forward_kinematics(urdf_path, zero_positions, kinematic_root_position)
    position_errors = []
    direction_passed = []
    for name in ALL_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = model.jnt_qposadr[joint_id]
        child_name = joints[name].child
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
        data.qpos[qadr] = 0.0
        mujoco.mj_forward(model, data)
        baseline = data.xpos[body_id].copy()
        data.qpos[qadr] = perturbation_rad
        mujoco.mj_forward(model, data)
        simulator_delta = data.xpos[body_id] - baseline
        perturbed = dict(zero_positions)
        perturbed[name] = perturbation_rad
        fk_delta = forward_kinematics(urdf_path, perturbed, kinematic_root_position)[child_name] - fk[child_name]
        position_errors.append(float(np.linalg.norm(fk[child_name] - baseline)))
        # A joint's own link may rotate about an origin with zero translation; examine its downstream wrist instead.
        direction_passed.append(bool(np.linalg.norm(simulator_delta - fk_delta) < 2e-5))
        data.qpos[qadr] = 0.0
    mujoco.mj_forward(model, data)
    # URDF left/right zero-pose geometry must be mirror-symmetric across y=0 at paired ankle and wrist links.
    symmetry_pairs = (
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    )
    symmetry_error = max(
        float(np.linalg.norm(fk[left] - fk[right] * np.array((1.0, -1.0, 1.0))))
        for left, right in symmetry_pairs
    )
    standing_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
    for name in ALL_JOINT_NAMES:
        if "hip_pitch" in name:
            standing_positions[name] = -0.28
        elif "knee" in name:
            standing_positions[name] = 0.79
        elif "ankle_pitch" in name:
            standing_positions[name] = -0.52
        elif "shoulder_pitch" in name:
            standing_positions[name] = 0.28
        elif "elbow" in name:
            standing_positions[name] = 0.52
    com, support_margin = static_com_and_support_margin(
        urdf_path, standing_positions, np.array(FIXED_BASE_ROOT_POSITION_W_M)
    )
    result = {
        "status": "passed",
        "total_mass_kg": sum(mass for mass, _ in inertials.values()),
        "links_with_collision_geometry": sum(count > 0 for count in collisions.values()),
        "links_without_collision_geometry": sorted(name for name, count in collisions.items() if count == 0),
        "max_urdf_to_mujoco_link_origin_error_m": max(position_errors),
        "joint_direction_checks_passed": all(direction_passed),
        "joint_direction_check_count": len(direction_passed),
        "left_right_mirror_error_m": symmetry_error,
        "static_com_w_m": com.tolist(),
        "static_lateral_support_margin_m": float(support_margin),
        "static_zmp_assumption": (
            "At static equilibrium with zero acceleration, ZMP equals CoM projection; this is not a dynamic ZMP "
            "validation."
        ),
    }
    if (
        result["max_urdf_to_mujoco_link_origin_error_m"] > 2e-5
        or not result["joint_direction_checks_passed"]
        or symmetry_error > 2e-5
        or support_margin <= 0.0
    ):
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    """Parse command line arguments and print the validation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("mjcf", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.urdf, args.mjcf), indent=2))


if __name__ == "__main__":
    main()
