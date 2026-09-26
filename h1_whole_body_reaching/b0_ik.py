# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""B0 single-arm damped-least-squares IK primitives.

The functions operate on Isaac Lab world-frame link positions and Jacobians.
They are deliberately separated from simulator setup so their numerical
contracts can be tested without launching Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
RIGHT_WRIST_LINK_NAME = "right_wrist_yaw_link"


def right_wrist_forward_kinematics(
    urdf_path: Path, joint_positions_rad: dict[str, float], root_position_w_m: np.ndarray
) -> np.ndarray:
    """Compute the right wrist link origin from the pinned URDF.

    Args:
        urdf_path: H1-2 URDF source path.
        joint_positions_rad: Articulation joint positions keyed by name [rad].
        root_position_w_m: Pelvis world position [m], shape ``[3]``.

    Returns:
        Right wrist-yaw-link origin in world coordinates [m], shape ``[3]``.
    """
    from h1_whole_body_reaching.phase_2_validation import forward_kinematics

    return forward_kinematics(urdf_path, joint_positions_rad, root_position_w_m)[RIGHT_WRIST_LINK_NAME]


@dataclass(frozen=True)
class B0IkCfg:
    """Numerical and motion limits for the B0 position-only IK loop.

    Args:
        damping: Damped-least-squares damping coefficient [m].
        maximum_joint_step_rad: Maximum Euclidean IK increment [rad].
        maximum_target_step_rad: Maximum command interpolation increment per
            control update [rad].
    """

    damping: float = 0.03
    maximum_joint_step_rad: float = 0.04
    maximum_target_step_rad: float = 0.02


def damped_least_squares_step(
    end_effector_position_w_m: "torch.Tensor",
    target_position_w_m: "torch.Tensor",
    position_jacobian_w_m_per_rad: "torch.Tensor",
    current_joint_position_rad: "torch.Tensor",
    lower_joint_limit_rad: "torch.Tensor",
    upper_joint_limit_rad: "torch.Tensor",
    cfg: B0IkCfg,
) -> "torch.Tensor":
    """Return one bounded, joint-limited position IK update.

    Args:
        end_effector_position_w_m: Current wrist-link origin [m], shape ``[N, 3]``.
        target_position_w_m: Desired wrist-link origin [m], shape ``[N, 3]``.
        position_jacobian_w_m_per_rad: World linear Jacobian, shape ``[N, 3, J]``.
        current_joint_position_rad: Current arm positions [rad], shape ``[N, J]``.
        lower_joint_limit_rad: Lower arm limits [rad], shape ``[N, J]`` or ``[J]``.
        upper_joint_limit_rad: Upper arm limits [rad], shape ``[N, J]`` or ``[J]``.
        cfg: Damping and update bounds.

    Returns:
        Next arm position targets [rad], shape ``[N, J]``.
    """
    import torch

    if cfg.damping <= 0.0 or cfg.maximum_joint_step_rad <= 0.0 or cfg.maximum_target_step_rad <= 0.0:
        raise ValueError("B0 IK damping and step limits must be positive.")
    if end_effector_position_w_m.shape != target_position_w_m.shape or end_effector_position_w_m.ndim != 2 or end_effector_position_w_m.shape[1] != 3:
        raise ValueError("End-effector and target positions must have shape [N, 3].")
    batch_size, joint_count = current_joint_position_rad.shape
    if position_jacobian_w_m_per_rad.shape != (batch_size, 3, joint_count):
        raise ValueError("Position Jacobian must have shape [N, 3, J].")
    if lower_joint_limit_rad.shape not in {(joint_count,), (batch_size, joint_count)}:
        raise ValueError("Lower joint limits must have shape [J] or [N, J].")
    if upper_joint_limit_rad.shape not in {(joint_count,), (batch_size, joint_count)}:
        raise ValueError("Upper joint limits must have shape [J] or [N, J].")
    if not all(torch.isfinite(value).all() for value in (end_effector_position_w_m, target_position_w_m, position_jacobian_w_m_per_rad, current_joint_position_rad)):
        raise ValueError("IK inputs must be finite.")
    if torch.any(lower_joint_limit_rad > upper_joint_limit_rad):
        raise ValueError("Lower joint limits must not exceed upper joint limits.")

    error_w_m = target_position_w_m - end_effector_position_w_m
    jacobian_transpose = position_jacobian_w_m_per_rad.transpose(1, 2)
    system = position_jacobian_w_m_per_rad @ jacobian_transpose
    identity = torch.eye(3, dtype=system.dtype, device=system.device).expand(batch_size, -1, -1)
    task_solution = torch.linalg.solve(system + cfg.damping**2 * identity, error_w_m.unsqueeze(-1))
    joint_increment_rad = (jacobian_transpose @ task_solution).squeeze(-1)
    increment_norm_rad = torch.linalg.vector_norm(joint_increment_rad, dim=-1, keepdim=True)
    joint_increment_rad *= (cfg.maximum_joint_step_rad / increment_norm_rad.clamp_min(cfg.maximum_joint_step_rad)).clamp_max(1.0)
    return (current_joint_position_rad + joint_increment_rad).clamp(lower_joint_limit_rad, upper_joint_limit_rad)


def interpolate_position_target(
    start_position_w_m: "torch.Tensor", target_position_w_m: "torch.Tensor", fraction: float
) -> "torch.Tensor":
    """Linearly interpolate a position target in the world frame.

    Args:
        start_position_w_m: Initial wrist position [m], shape ``[N, 3]``.
        target_position_w_m: Final wrist target [m], shape ``[N, 3]``.
        fraction: Interpolation progress in ``[0, 1]``.

    Returns:
        Interpolated wrist target [m], shape ``[N, 3]``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Interpolation fraction must be in [0, 1].")
    if start_position_w_m.shape != target_position_w_m.shape or start_position_w_m.ndim != 2 or start_position_w_m.shape[1] != 3:
        raise ValueError("Position targets must have shape [N, 3].")
    return start_position_w_m + fraction * (target_position_w_m - start_position_w_m)


def smooth_joint_position_target(
    previous_target_rad: "torch.Tensor", desired_target_rad: "torch.Tensor", maximum_step_rad: float
) -> "torch.Tensor":
    """Bound the per-control-update change in joint position targets.

    Args:
        previous_target_rad: Previous position target [rad], shape ``[N, J]``.
        desired_target_rad: New desired target [rad], shape ``[N, J]``.
        maximum_step_rad: Maximum absolute per-joint change [rad].

    Returns:
        Smoothed position target [rad], shape ``[N, J]``.
    """
    if maximum_step_rad <= 0.0:
        raise ValueError("maximum_step_rad must be positive.")
    if previous_target_rad.shape != desired_target_rad.shape or previous_target_rad.ndim != 2:
        raise ValueError("Joint targets must share shape [N, J].")
    return previous_target_rad + (desired_target_rad - previous_target_rad).clamp(-maximum_step_rad, maximum_step_rad)
