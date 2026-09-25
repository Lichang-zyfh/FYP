# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen H1-2 action-interface definitions."""

from __future__ import annotations

from collections.abc import Sequence


ALL_JOINT_NAMES = (
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
WRIST_JOINT_NAMES = tuple(name for name in ALL_JOINT_NAMES if "wrist" in name)
POLICY_JOINT_NAMES = tuple(name for name in ALL_JOINT_NAMES if name not in WRIST_JOINT_NAMES)


def resolve_joint_ids(actual_joint_names: Sequence[str], requested_joint_names: Sequence[str]) -> list[int]:
    """Resolve an ordered joint-name contract to indices.

    Args:
        actual_joint_names: Joint names exposed by the loaded articulation.
        requested_joint_names: Joint names in the required public order.

    Returns:
        Joint indices in the order of ``requested_joint_names``.

    Raises:
        ValueError: If the articulation contains duplicate names or a required joint is missing.
    """
    if len(set(actual_joint_names)) != len(actual_joint_names):
        raise ValueError("The articulation contains duplicate joint names.")
    name_to_id = {name: index for index, name in enumerate(actual_joint_names)}
    missing = [name for name in requested_joint_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Required joints are missing: {missing}")
    return [name_to_id[name] for name in requested_joint_names]


def expand_policy_action(
    policy_action: Sequence[float], nominal_joint_positions: Sequence[float], action_scale: float
) -> list[float]:
    """Expand a 21-value policy action into all 27 H1-2 joint targets.

    The six wrist targets remain at their nominal positions. All quantities are joint
    positions or offsets [rad].

    Args:
        policy_action: Normalized policy action in ``POLICY_JOINT_NAMES`` order.
        nominal_joint_positions: Nominal targets in ``ALL_JOINT_NAMES`` order [rad].
        action_scale: Position-offset scale [rad].

    Returns:
        Joint targets in ``ALL_JOINT_NAMES`` order [rad].

    Raises:
        ValueError: If an input has the wrong length or an action is outside ``[-1, 1]``.
    """
    if len(policy_action) != len(POLICY_JOINT_NAMES):
        raise ValueError(f"Expected {len(POLICY_JOINT_NAMES)} policy actions, got {len(policy_action)}.")
    if len(nominal_joint_positions) != len(ALL_JOINT_NAMES):
        raise ValueError(f"Expected {len(ALL_JOINT_NAMES)} nominal positions, got {len(nominal_joint_positions)}.")
    if any(value < -1.0 or value > 1.0 for value in policy_action):
        raise ValueError("Policy actions must be within [-1, 1].")

    policy_values = dict(zip(POLICY_JOINT_NAMES, policy_action, strict=True))
    return [
        nominal + action_scale * policy_values[name] if name in policy_values else nominal
        for name, nominal in zip(ALL_JOINT_NAMES, nominal_joint_positions, strict=True)
    ]
