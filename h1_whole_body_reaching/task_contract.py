# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 right-hand target, observation, and termination contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from random import Random

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, POLICY_JOINT_NAMES

RIGHT_END_EFFECTOR_BODY_NAME = "right_wrist_yaw_link"
FIXED_RIGHT_HAND_TARGET_POSITION_W_M = (0.0843, -0.2163, 0.9063)
SCRIPTED_RIGHT_SHOULDER_PITCH_ACTION = -0.5


@dataclass
class ReachingTaskCfg:
    """Phase 3 vectorized reaching-task parameters.

    The narrow B0 workspace is outside the initial 0.05 m success radius and
    its representative lower and upper corners passed free-base IK/PD probes.
    This is not a general whole-body workspace claim.
    """

    target_lower_w_m: tuple[float, float, float] = (0.1215, -0.221, 0.909)
    target_upper_w_m: tuple[float, float, float] = (0.1225, -0.219, 0.911)
    success_tolerance_m: float = 0.05
    success_hold_duration_s: float = 0.5
    reaching_reward_scale: float = 1.0
    success_reward: float = 2.0
    free_base_fall_height_m: float = 0.75
    free_base_min_upright_cosine: float = 0.90
    free_base_tilt_grace_duration_s: float = 0.25
    free_base_min_foot_contact_force_n: float = 20.0
    free_base_contact_grace_duration_s: float = 0.10


@dataclass
class StandingTaskCfg:
    """Reward configuration for the Phase 6 fixed-pose standing task.

    The standing task intentionally contains no target, end-effector, or
    reaching-success term. All weights are unitless reward multipliers.
    """

    upright_reward_weight: float = 1.0
    base_height_reward_weight: float = 1.0
    foot_contact_reward_weight: float = 0.5
    posture_reward_weight: float = 0.5
    action_smoothness_reward_weight: float = 0.05
    fall_penalty: float = 5.0
    posture_error_scale_rad: float = 0.25
    base_height_error_scale_m: float = 0.10


def validate_standing_task_cfg(task_cfg: StandingTaskCfg) -> None:
    """Validate Phase 6 standing reward parameters."""
    weights = (
        task_cfg.upright_reward_weight,
        task_cfg.base_height_reward_weight,
        task_cfg.foot_contact_reward_weight,
        task_cfg.posture_reward_weight,
        task_cfg.action_smoothness_reward_weight,
        task_cfg.fall_penalty,
    )
    if any(not isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Standing reward weights must be finite and non-negative.")
    if task_cfg.posture_error_scale_rad <= 0.0 or task_cfg.base_height_error_scale_m <= 0.0:
        raise ValueError("Standing reward error scales must be positive.")


@dataclass(frozen=True)
class FixedRightHandTargetCfg:
    """Fixed target and termination limits for the first reaching probe.

    Attributes:
        position_w_m: Right-hand target position in world coordinates [m].
        success_tolerance_m: Maximum Euclidean hand-target error for success [m].
        episode_length_steps: Maximum scripted-control steps before timeout.
    """

    position_w_m: tuple[float, float, float] = FIXED_RIGHT_HAND_TARGET_POSITION_W_M
    success_tolerance_m: float = 0.01
    episode_length_steps: int = 200


@dataclass(frozen=True)
class ResetRandomizationCfg:
    """Deterministic policy-joint reset randomization for the first probe.

    Attributes:
        policy_joint_position_offset_bound_rad: Symmetric position-offset bound
            applied only to policy joints [rad].
    """

    policy_joint_position_offset_bound_rad: float = 0.01


@dataclass(frozen=True)
class TargetTermination:
    """Termination flags for the fixed-target reaching probe."""

    success: bool
    time_out: bool


@dataclass(frozen=True)
class TaskTermination:
    """Explicit termination flags for the fixed-base reaching prototype.

    The fixed pelvis makes fall detection inapplicable at this stage. The first
    safety contract instead rejects non-finite state and positions outside the
    imported hard joint limits.
    """

    success: bool
    time_out: bool
    invalid_state: bool
    joint_limit_violation: bool


def validate_reaching_task_cfg(task_cfg: ReachingTaskCfg) -> None:
    """Validate the bounded Phase 3 target-sampling contract."""
    if any(lower >= upper for lower, upper in zip(task_cfg.target_lower_w_m, task_cfg.target_upper_w_m, strict=True)):
        raise ValueError("Each target lower bound must be smaller than its upper bound.")
    if task_cfg.success_tolerance_m <= 0.0 or task_cfg.success_hold_duration_s <= 0.0:
        raise ValueError("Success tolerance and hold duration must be positive.")
    if task_cfg.free_base_fall_height_m <= 0.0:
        raise ValueError("Free-base fall height must be positive.")
    if not -1.0 <= task_cfg.free_base_min_upright_cosine <= 1.0:
        raise ValueError("Free-base upright cosine must be in [-1, 1].")
    if task_cfg.free_base_tilt_grace_duration_s < 0.0:
        raise ValueError("Free-base tilt grace duration must be non-negative.")
    if task_cfg.free_base_min_foot_contact_force_n <= 0.0:
        raise ValueError("Free-base foot-contact force threshold must be positive.")
    if task_cfg.free_base_contact_grace_duration_s < 0.0:
        raise ValueError("Free-base contact grace duration must be non-negative.")


def target_box_corners_and_center(task_cfg: ReachingTaskCfg) -> tuple[tuple[float, float, float], ...]:
    """Return the eight target-box corners followed by its center [m]."""
    validate_reaching_task_cfg(task_cfg)
    lower = task_cfg.target_lower_w_m
    upper = task_cfg.target_upper_w_m
    corners = tuple((x, y, z) for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2]))
    center = tuple((low + high) / 2.0 for low, high in zip(lower, upper, strict=True))
    return (*corners, center)


def target_box_initial_errors_m(
    end_effector_position_w_m: Sequence[float], task_cfg: ReachingTaskCfg
) -> tuple[float, ...]:
    """Return initial wrist errors for every target-box corner and center [m]."""
    if len(end_effector_position_w_m) != 3 or not all(isfinite(value) for value in end_effector_position_w_m):
        raise ValueError("End-effector position must contain three finite coordinates.")
    return tuple(
        sqrt(sum((target_coordinate - wrist_coordinate) ** 2 for target_coordinate, wrist_coordinate in zip(target, end_effector_position_w_m, strict=True)))
        for target in target_box_corners_and_center(task_cfg)
    )


def reaching_reward(target_error_m: float, task_cfg: ReachingTaskCfg, success: bool) -> float:
    """Return bounded dense reaching reward plus a held-success bonus."""
    if not isfinite(target_error_m) or target_error_m < 0.0:
        raise ValueError("Target error must be finite and non-negative.")
    return task_cfg.reaching_reward_scale * (1.0 - min(target_error_m, 1.0)) + task_cfg.success_reward * success


def make_scripted_target_action() -> list[float]:
    """Create the bounded 21-value action for the fixed right-hand target.

    Returns:
        Normalized policy action in ``POLICY_JOINT_NAMES`` order.
    """
    action = [0.0] * len(POLICY_JOINT_NAMES)
    action[POLICY_JOINT_NAMES.index("right_shoulder_pitch_joint")] = SCRIPTED_RIGHT_SHOULDER_PITCH_ACTION
    return action


def make_reset_policy_joint_offsets(
    seed: int, reset_cfg: ResetRandomizationCfg = ResetRandomizationCfg()
) -> list[float]:
    """Sample deterministic policy-joint reset offsets.

    Args:
        seed: Pseudorandom generator seed.
        reset_cfg: Symmetric policy-joint offset bound.

    Returns:
        One offset per policy joint in ``POLICY_JOINT_NAMES`` order [rad].
    """
    bound = reset_cfg.policy_joint_position_offset_bound_rad
    if not isfinite(bound) or bound < 0.0:
        raise ValueError("Reset offset bound must be finite and non-negative.")
    generator = Random(seed)
    return [generator.uniform(-bound, bound) for _ in POLICY_JOINT_NAMES]


def apply_reset_policy_joint_offsets(
    nominal_joint_positions_rad: Sequence[float], policy_joint_offsets_rad: Sequence[float]
) -> list[float]:
    """Apply policy-only offsets while preserving the six wrist reset targets.

    Args:
        nominal_joint_positions_rad: Nominal articulation targets in
            ``ALL_JOINT_NAMES`` order [m or rad, depending on joint type].
        policy_joint_offsets_rad: Policy-joint offsets in ``POLICY_JOINT_NAMES``
            order [rad].

    Returns:
        Randomized articulation targets in ``ALL_JOINT_NAMES`` order [m or rad,
        depending on joint type].
    """
    if len(nominal_joint_positions_rad) != len(ALL_JOINT_NAMES):
        raise ValueError("Expected one nominal target for each actuated joint.")
    if len(policy_joint_offsets_rad) != len(POLICY_JOINT_NAMES):
        raise ValueError(f"Expected {len(POLICY_JOINT_NAMES)} policy-joint reset offsets.")

    targets = list(nominal_joint_positions_rad)
    for name, offset in zip(POLICY_JOINT_NAMES, policy_joint_offsets_rad, strict=True):
        targets_index = ALL_JOINT_NAMES.index(name)
        targets[targets_index] += offset
    return targets


def make_target_observation(
    end_effector_position_w_m: tuple[float, float, float], target_cfg: FixedRightHandTargetCfg
) -> tuple[float, float, float, float, float, float]:
    """Build the minimal reaching observation.

    Args:
        end_effector_position_w_m: Right wrist position in world coordinates [m].
        target_cfg: Fixed-target configuration.

    Returns:
        Concatenated right-wrist position and target-relative error [m].
    """
    if len(end_effector_position_w_m) != 3:
        raise ValueError("End-effector position must have three coordinates.")
    coordinate_pairs = zip(end_effector_position_w_m, target_cfg.position_w_m, strict=True)
    error_w_m = tuple(target - position for position, target in coordinate_pairs)
    return (*end_effector_position_w_m, *error_w_m)


def target_error_norm_m(observation: tuple[float, float, float, float, float, float]) -> float:
    """Return the Euclidean hand-target error from a reaching observation [m]."""
    if len(observation) != 6:
        raise ValueError("Target observation must contain six values.")
    return sqrt(sum(component * component for component in observation[3:]))


def evaluate_target_termination(
    target_error_m: float, step_count: int, target_cfg: FixedRightHandTargetCfg
) -> TargetTermination:
    """Evaluate fixed-target success and timeout conditions.

    Args:
        target_error_m: Euclidean right-hand target error [m].
        step_count: Completed scripted-control steps.
        target_cfg: Fixed-target configuration.

    Returns:
        Success and timeout flags. Success takes precedence at the final step.
    """
    if step_count < 0:
        raise ValueError("Step count must not be negative.")
    success = target_error_m <= target_cfg.success_tolerance_m
    return TargetTermination(success=success, time_out=not success and step_count >= target_cfg.episode_length_steps)


def evaluate_task_termination(
    target_error_m: float,
    step_count: int,
    target_cfg: FixedRightHandTargetCfg,
    joint_positions_rad: Sequence[float],
    joint_lower_limits_rad: Sequence[float],
    joint_upper_limits_rad: Sequence[float],
) -> TaskTermination:
    """Evaluate the first explicit safety and task termination set.

    Args:
        target_error_m: Euclidean right-hand target error [m].
        step_count: Completed scripted-control steps.
        target_cfg: Fixed-target configuration.
        joint_positions_rad: Current articulation positions [m or rad, depending
            on joint type].
        joint_lower_limits_rad: Imported hard lower joint limits [m or rad,
            depending on joint type].
        joint_upper_limits_rad: Imported hard upper joint limits [m or rad,
            depending on joint type].

    Returns:
        Success, timeout, invalid-state, and hard-joint-limit termination flags.
    """
    if step_count < 0:
        raise ValueError("Step count must not be negative.")
    if not (len(joint_positions_rad) == len(joint_lower_limits_rad) == len(joint_upper_limits_rad)):
        raise ValueError("Joint positions and limits must have identical lengths.")

    invalid_state = not isfinite(target_error_m) or not all(isfinite(value) for value in joint_positions_rad)
    joint_limit_violation = any(
        position < lower or position > upper
        for position, lower, upper in zip(
            joint_positions_rad, joint_lower_limits_rad, joint_upper_limits_rad, strict=True
        )
    )
    success = not invalid_state and not joint_limit_violation and target_error_m <= target_cfg.success_tolerance_m
    time_out = (
        not success
        and not invalid_state
        and not joint_limit_violation
        and step_count >= target_cfg.episode_length_steps
    )
    return TaskTermination(
        success=success,
        time_out=time_out,
        invalid_state=invalid_state,
        joint_limit_violation=joint_limit_violation,
    )
