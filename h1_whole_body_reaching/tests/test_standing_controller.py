# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the bounded free-base standing baseline."""

import pytest
import torch

from h1_whole_body_reaching.standing_controller import StandingStabilizerCfg, apply_standing_stabilizer


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
)


def test_sagittal_stabilizer_coordinates_symmetric_leg_targets() -> None:
    """A forward tilt applies the configured bounded correction to both legs."""
    targets = torch.tensor([[-0.28, -0.28, 0.79, 0.79, -0.52, -0.52, 0.0, 0.0, 0.0, 0.0, 0.28, 0.28]])
    corrected = apply_standing_stabilizer(
        targets,
        JOINT_NAMES,
        torch.tensor([[0.10, 0.0, 0.995]]),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        StandingStabilizerCfg(pitch_kp=2.0, hip_pitch_ratio=-0.5, knee_ratio=0.25, max_target_offset_rad=0.15),
    )

    torch.testing.assert_close(
        corrected, torch.tensor([[-0.355, -0.355, 0.8275, 0.8275, -0.37, -0.37, 0.0, 0.0, 0.0, 0.0, 0.28, 0.28]])
    )


def test_sagittal_stabilizer_clamps_and_rejects_missing_joint() -> None:
    """Corrections remain bounded and the two ankle joints are required."""
    targets = torch.zeros((1, 12))
    corrected = apply_standing_stabilizer(
        targets,
        JOINT_NAMES,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        StandingStabilizerCfg(max_target_offset_rad=0.05),
    )
    assert corrected[0, 4].item() == pytest.approx(0.05)
    assert corrected[0, 5].item() == pytest.approx(0.05)
    with pytest.raises(ValueError, match="left_knee_joint"):
        apply_standing_stabilizer(
            targets[:, :1],
            JOINT_NAMES[:2],
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            StandingStabilizerCfg(),
        )


def test_sagittal_stabilizer_resists_forward_root_drift() -> None:
    """Forward position and velocity produce a bounded reverse correction."""
    corrected = apply_standing_stabilizer(
        torch.zeros((1, 12)),
        JOINT_NAMES,
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[0.10, 0.0, 1.0]]),
        torch.tensor([[0.02, 0.0, 1.0]]),
        torch.tensor([[0.30, 0.0, 0.0]]),
        StandingStabilizerCfg(root_position_kp=1.0, root_velocity_kd=0.5, max_target_offset_rad=0.20),
    )
    assert corrected[0, 4].item() == pytest.approx(-0.20)
    assert corrected[0, 5].item() == pytest.approx(-0.20)


def test_stabilizer_uses_only_valid_support_center_feedback() -> None:
    """A valid support center supplies a bounded pelvis-to-support correction."""
    targets = torch.zeros((2, 12))
    corrected = apply_standing_stabilizer(
        targets,
        JOINT_NAMES,
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        torch.tensor([[0.10, 0.0, 1.0], [0.10, 0.0, 1.0]]),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        StandingStabilizerCfg(support_position_kp=-2.0, max_target_offset_rad=0.10),
        torch.tensor([[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(corrected[:, 4], torch.tensor([0.10, 0.0]))
    torch.testing.assert_close(corrected[:, 5], torch.tensor([0.10, 0.0]))


def test_stabilizer_uses_capture_point_relative_to_support() -> None:
    """Forward velocity advances the capture point and is corrected only with valid support."""
    corrected = apply_standing_stabilizer(
        torch.zeros((1, 12)),
        JOINT_NAMES,
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[0.02, 0.0, 1.0]]),
        torch.zeros((1, 3)),
        torch.tensor([[0.30, 0.0, 0.0]]),
        StandingStabilizerCfg(capture_point_kp=-1.0, max_target_offset_rad=0.20),
        torch.zeros((1, 3)),
        torch.tensor([True]),
    )
    expected_capture_error = 0.02 + 0.30 / (9.81 / 1.0) ** 0.5
    assert corrected[0, 4].item() == pytest.approx(expected_capture_error)
    assert corrected[0, 5].item() == pytest.approx(expected_capture_error)


def test_stabilizer_coordinates_symmetric_shoulder_pitch_targets() -> None:
    """Pitch correction can move both shoulders without changing leg symmetry."""
    corrected = apply_standing_stabilizer(
        torch.zeros((1, 12)),
        JOINT_NAMES,
        torch.tensor([[0.05, 0.0, 0.999]]),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        StandingStabilizerCfg(pitch_kp=1.0, shoulder_pitch_ratio=2.0),
    )
    assert corrected[0, 10].item() == pytest.approx(0.10)
    assert corrected[0, 11].item() == pytest.approx(0.10)


def test_stabilizer_coordinates_symmetric_ankle_roll_targets() -> None:
    """Lateral tilt and roll velocity apply equal bounded ankle-roll targets."""
    corrected = apply_standing_stabilizer(
        torch.zeros((1, 12)),
        JOINT_NAMES,
        torch.tensor([[0.0, 0.08, 0.997]]),
        torch.tensor([[0.20, 0.0, 0.0]]),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        StandingStabilizerCfg(roll_kp=1.0, roll_kd=0.2, max_roll_target_offset_rad=0.10),
    )
    assert corrected[0, 6].item() == pytest.approx(0.10)
    assert corrected[0, 7].item() == pytest.approx(0.10)


def test_stabilizer_uses_lateral_capture_point_and_distributes_roll_to_hips() -> None:
    """Lateral support velocity affects roll targets and can coordinate both hips."""
    corrected = apply_standing_stabilizer(
        torch.zeros((1, 12)),
        JOINT_NAMES,
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.02, 1.0]]),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.30, 0.0]]),
        StandingStabilizerCfg(
            lateral_capture_point_kp=-1.0,
            left_hip_roll_ratio=0.5,
            right_hip_roll_ratio=-0.5,
            max_roll_target_offset_rad=0.20,
        ),
        torch.zeros((1, 3)),
        torch.tensor([True]),
    )
    expected_roll = -(0.02 + 0.30 / (9.81 / 1.0) ** 0.5)
    assert corrected[0, 6].item() == pytest.approx(expected_roll)
    assert corrected[0, 7].item() == pytest.approx(expected_roll)
    assert corrected[0, 8].item() == pytest.approx(0.5 * expected_roll)
    assert corrected[0, 9].item() == pytest.approx(-0.5 * expected_roll)
