# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded joint feedback used to validate the H1-2 free-base standing gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class StandingStabilizerCfg:
    """Gains for the free-base standing diagnostic.

    The pitch terms map the horizontal component of the pelvis up axis and the
    world-frame pitch angular velocity to equal left/right ankle-pitch target
    offsets. Root horizontal position and velocity feedback can be added to
    that signal to resist forward drift. A separate roll signal maps pelvis
    lateral tilt and roll velocity to equal left/right ankle-roll offsets.
    Optional lateral support and capture-point feedback augment that roll
    signal, which can also be distributed independently to the hip-roll joints. The
    corrections do not change the policy action interface.

    Args:
        pitch_kp: Up-axis proportional gain [rad/rad].
        pitch_kd: Angular-velocity derivative gain [s].
        hip_pitch_ratio: Hip-pitch correction relative to ankle correction [-].
        knee_ratio: Knee correction relative to ankle correction [-].
        root_position_kp: Forward root-position error gain [rad/m].
        root_velocity_kd: Forward root-velocity gain [s/m].
        support_position_kp: Pelvis-to-support-center forward-error gain [rad/m].
        capture_point_kp: Linear-inverted-pendulum capture-point-to-support gain [rad/m].
        shoulder_pitch_ratio: Shoulder-pitch correction relative to ankle correction [-].
        roll_kp: Pelvis up-axis lateral-component gain [rad/rad].
        roll_kd: Roll angular-velocity gain [s].
        left_hip_roll_ratio: Left hip-roll correction relative to ankle-roll correction [-].
        right_hip_roll_ratio: Right hip-roll correction relative to ankle-roll correction [-].
        lateral_support_position_kp: Pelvis-to-support-center lateral-error gain [rad/m].
        lateral_capture_point_kp: Lateral capture-point-to-support gain [rad/m].
        max_target_offset_rad: Per-ankle target correction magnitude [rad].
        max_roll_target_offset_rad: Per-ankle roll correction magnitude [rad].
    """

    pitch_kp: float = 1.0
    pitch_kd: float = 0.15
    hip_pitch_ratio: float = 0.0
    knee_ratio: float = 0.0
    root_position_kp: float = 0.0
    root_velocity_kd: float = 0.0
    support_position_kp: float = 0.0
    capture_point_kp: float = 0.0
    shoulder_pitch_ratio: float = 0.0
    roll_kp: float = 0.0
    roll_kd: float = 0.0
    left_hip_roll_ratio: float = 0.0
    right_hip_roll_ratio: float = 0.0
    lateral_support_position_kp: float = 0.0
    lateral_capture_point_kp: float = 0.0
    max_target_offset_rad: float = 0.20
    max_roll_target_offset_rad: float = 0.10


def apply_standing_stabilizer(
    nominal_targets: "torch.Tensor",
    joint_names: tuple[str, ...],
    pelvis_up_axis_w: "torch.Tensor",
    pelvis_angular_velocity_w_radps: "torch.Tensor",
    pelvis_position_w_m: "torch.Tensor",
    target_pelvis_position_w_m: "torch.Tensor",
    pelvis_linear_velocity_w_mps: "torch.Tensor",
    cfg: StandingStabilizerCfg,
    support_center_w_m: "torch.Tensor | None" = None,
    support_center_valid: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Return nominal targets with bounded pitch and roll corrections.

    Args:
        nominal_targets: Nominal position targets [rad], shape ``[N, J]``.
        joint_names: Names corresponding to the last dimension of ``nominal_targets``.
        pelvis_up_axis_w: Pelvis local positive-z axis in world coordinates, shape ``[N, 3]``.
        pelvis_angular_velocity_w_radps: Pelvis angular velocity in world coordinates [rad/s], shape ``[N, 3]``.
        pelvis_position_w_m: Pelvis position in world coordinates [m], shape ``[N, 3]``.
        target_pelvis_position_w_m: Reference pelvis position in world coordinates [m], shape ``[N, 3]``.
        pelvis_linear_velocity_w_mps: Pelvis linear velocity in world coordinates [m/s], shape ``[N, 3]``.
        cfg: Bounded ankle-stabilizer gains.
        support_center_w_m: Force-weighted ground-contact center [m], shape ``[N, 3]``. When omitted,
            support feedback is disabled.
        support_center_valid: Whether each support center is valid, shape ``[N]``. Invalid entries
            contribute no support feedback.

    Returns:
        Corrected joint position targets [rad], shape ``[N, J]``.

    Raises:
        ValueError: If a required leg joint is absent or tensor shapes are incompatible.
    """
    required_names = (
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
    missing = [name for name in required_names if name not in joint_names]
    if missing:
        raise ValueError(f"Standing stabilizer requires joints: {missing}.")
    if nominal_targets.ndim != 2 or pelvis_up_axis_w.shape != (nominal_targets.shape[0], 3):
        raise ValueError("Expected targets [N, J] and pelvis up-axis [N, 3].")
    if pelvis_angular_velocity_w_radps.shape != pelvis_up_axis_w.shape:
        raise ValueError("Pelvis angular velocity must have shape [N, 3].")
    if pelvis_position_w_m.shape != pelvis_up_axis_w.shape:
        raise ValueError("Pelvis position must have shape [N, 3].")
    if target_pelvis_position_w_m.shape != pelvis_up_axis_w.shape:
        raise ValueError("Target pelvis position must have shape [N, 3].")
    if pelvis_linear_velocity_w_mps.shape != pelvis_up_axis_w.shape:
        raise ValueError("Pelvis linear velocity must have shape [N, 3].")
    if support_center_w_m is not None and support_center_w_m.shape != pelvis_up_axis_w.shape:
        raise ValueError("Support center must have shape [N, 3].")
    if support_center_valid is not None and support_center_valid.shape != (nominal_targets.shape[0],):
        raise ValueError("Support-center validity must have shape [N].")
    if support_center_valid is not None and support_center_w_m is None:
        raise ValueError("Support-center validity requires support-center positions.")
    if cfg.max_target_offset_rad <= 0.0:
        raise ValueError("max_target_offset_rad must be positive.")
    if cfg.max_roll_target_offset_rad <= 0.0:
        raise ValueError("max_roll_target_offset_rad must be positive.")

    support_position_offset_rad = 0.0
    capture_point_offset_rad = 0.0
    lateral_support_position_offset_rad = 0.0
    lateral_capture_point_offset_rad = 0.0
    if support_center_w_m is not None:
        support_error_m = pelvis_position_w_m[:, 0] - support_center_w_m[:, 0]
        capture_point_error_m = support_error_m + pelvis_linear_velocity_w_mps[:, 0] / (
            (9.81 / pelvis_position_w_m[:, 2].clamp_min(0.10)).sqrt()
        )
        if support_center_valid is not None:
            support_error_m = support_error_m * support_center_valid.to(dtype=support_error_m.dtype)
            capture_point_error_m = capture_point_error_m * support_center_valid.to(dtype=capture_point_error_m.dtype)
        support_position_offset_rad = -cfg.support_position_kp * support_error_m
        capture_point_offset_rad = -cfg.capture_point_kp * capture_point_error_m
        lateral_support_error_m = pelvis_position_w_m[:, 1] - support_center_w_m[:, 1]
        lateral_capture_point_error_m = lateral_support_error_m + pelvis_linear_velocity_w_mps[:, 1] / (
            (9.81 / pelvis_position_w_m[:, 2].clamp_min(0.10)).sqrt()
        )
        if support_center_valid is not None:
            lateral_support_error_m = lateral_support_error_m * support_center_valid.to(dtype=lateral_support_error_m.dtype)
            lateral_capture_point_error_m = lateral_capture_point_error_m * support_center_valid.to(
                dtype=lateral_capture_point_error_m.dtype
            )
        lateral_support_position_offset_rad = cfg.lateral_support_position_kp * lateral_support_error_m
        lateral_capture_point_offset_rad = cfg.lateral_capture_point_kp * lateral_capture_point_error_m
    pitch_offset_rad = (
        cfg.pitch_kp * pelvis_up_axis_w[:, 0]
        + cfg.pitch_kd * pelvis_angular_velocity_w_radps[:, 1]
        - cfg.root_position_kp * (pelvis_position_w_m[:, 0] - target_pelvis_position_w_m[:, 0])
        - cfg.root_velocity_kd * pelvis_linear_velocity_w_mps[:, 0]
        + support_position_offset_rad
        + capture_point_offset_rad
    ).clamp(-cfg.max_target_offset_rad, cfg.max_target_offset_rad)
    targets = nominal_targets.clone()
    for joint_name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        targets[:, joint_names.index(joint_name)] += pitch_offset_rad
    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        targets[:, joint_names.index(joint_name)] += cfg.hip_pitch_ratio * pitch_offset_rad
    for joint_name in ("left_knee_joint", "right_knee_joint"):
        targets[:, joint_names.index(joint_name)] += cfg.knee_ratio * pitch_offset_rad
    for joint_name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        targets[:, joint_names.index(joint_name)] += cfg.shoulder_pitch_ratio * pitch_offset_rad
    roll_offset_rad = (
        cfg.roll_kp * pelvis_up_axis_w[:, 1]
        + cfg.roll_kd * pelvis_angular_velocity_w_radps[:, 0]
        + lateral_support_position_offset_rad
        + lateral_capture_point_offset_rad
    ).clamp(-cfg.max_roll_target_offset_rad, cfg.max_roll_target_offset_rad)
    for joint_name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
        targets[:, joint_names.index(joint_name)] += roll_offset_rad
    targets[:, joint_names.index("left_hip_roll_joint")] += cfg.left_hip_roll_ratio * roll_offset_rad
    targets[:, joint_names.index("right_hip_roll_joint")] += cfg.right_hip_roll_ratio * roll_offset_rad
    return targets


# Backward-compatible aliases for the initial sagittal-only diagnostic API.
SagittalStabilizerCfg = StandingStabilizerCfg
apply_sagittal_stabilizer = apply_standing_stabilizer
