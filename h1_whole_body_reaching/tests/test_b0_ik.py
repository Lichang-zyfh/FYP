# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical contracts for the B0 single-arm IK baseline."""

from pathlib import Path

import numpy as np
import pytest
import torch

from h1_whole_body_reaching.b0_ik import B0IkCfg, damped_least_squares_step, interpolate_position_target, right_wrist_forward_kinematics, smooth_joint_position_target
from h1_whole_body_reaching.phase_2_validation import forward_kinematics
from h1_whole_body_reaching.reaching_env_cfg import make_b0_reaching_env_cfg


def test_damped_least_squares_reduces_a_reachable_position_error() -> None:
    """A full-rank Cartesian Jacobian moves the wrist toward a reachable target."""
    current = torch.zeros((1, 3))
    target = torch.tensor([[0.06, -0.03, 0.02]])
    next_target = damped_least_squares_step(
        torch.zeros((1, 3)), target, torch.eye(3).unsqueeze(0), current,
        torch.full((3,), -1.0), torch.full((3,), 1.0), B0IkCfg(damping=0.01, maximum_joint_step_rad=0.2),
    )
    assert torch.linalg.vector_norm(target - next_target).item() < torch.linalg.vector_norm(target - current).item()


def test_damped_least_squares_clamps_joint_limits_and_increment() -> None:
    """Singular or large-error updates remain bounded and within limits."""
    next_target = damped_least_squares_step(
        torch.zeros((1, 3)), torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([[[1.0], [0.0], [0.0]]]), torch.zeros((1, 1)),
        torch.tensor([-0.02]), torch.tensor([0.02]), B0IkCfg(damping=0.1, maximum_joint_step_rad=0.5),
    )
    torch.testing.assert_close(next_target, torch.tensor([[0.02]]))


def test_target_interpolation_and_smoothing_are_bounded() -> None:
    """Both task-space and joint-space commands advance by their configured bounds."""
    interpolated = interpolate_position_target(torch.zeros((1, 3)), torch.tensor([[1.0, 2.0, 3.0]]), 0.25)
    torch.testing.assert_close(interpolated, torch.tensor([[0.25, 0.5, 0.75]]))
    smoothed = smooth_joint_position_target(torch.zeros((1, 2)), torch.tensor([[0.1, -0.1]]), 0.03)
    torch.testing.assert_close(smoothed, torch.tensor([[0.03, -0.03]]))
    with pytest.raises(ValueError, match="fraction"):
        interpolate_position_target(torch.zeros((1, 3)), torch.zeros((1, 3)), 1.1)


def test_right_wrist_fk_uses_the_existing_urdf_contract() -> None:
    """B0 FK is explicitly pinned to the independently validated URDF routine."""
    urdf_path = Path("assets/h1_2/derived/h1_2_handless_free_base.urdf")
    joint_positions = {"right_shoulder_pitch_joint": 0.28, "right_elbow_joint": 0.52}
    expected = forward_kinematics(urdf_path, joint_positions, np.array((0.0, 0.0, 0.9625)))["right_wrist_yaw_link"]
    np.testing.assert_allclose(right_wrist_forward_kinematics(urdf_path, joint_positions, np.array((0.0, 0.0, 0.9625))), expected)


def test_b0_uses_one_explicit_free_base_standing_configuration() -> None:
    """B0 rollout and persisted evaluation share the same standing contract."""
    cfg = make_b0_reaching_env_cfg(Path("assets/h1_2/derived/h1_2_handless_free_base.urdf"), 1, "cpu")
    assert not cfg.fixed_base
    assert cfg.free_base_shoulder_pitch_offset_rad == 0.0
    assert cfg.free_base_stabilizer["pitch_kp"] == 1.0
    assert cfg.free_base_stabilizer["root_position_kp"] == -5.0
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.0
