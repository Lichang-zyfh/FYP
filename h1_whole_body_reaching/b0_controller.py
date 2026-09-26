# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful action generator for the B0 single-arm IK/PD baseline."""

from __future__ import annotations

from typing import Any

import torch

from h1_whole_body_reaching.b0_ik import B0IkCfg, RIGHT_ARM_JOINT_NAMES, damped_least_squares_step, smooth_joint_position_target
from h1_whole_body_reaching.interfaces import POLICY_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.robot_cfg import ACTION_SCALE_RAD
from h1_whole_body_reaching.task_contract import RIGHT_END_EFFECTOR_BODY_NAME


class B0ActionGenerator:
    """Generate smooth 21-dimensional policy actions from a wrist target.

    The generator uses the simulator's world-frame wrist Jacobian and leaves
    all non-right-arm policy coordinates at zero for the standing controller.
    """

    def __init__(self, robot: Any, device: str, *, ik_cfg: B0IkCfg | None = None) -> None:
        """Initialize joint/body indices and numerical limits.

        Args:
            robot: Isaac Lab H1 articulation instance.
            device: Torch device used by the associated environment.
            ik_cfg: Optional B0 numerical configuration.
        """
        self._robot = robot
        self._device = device
        self._arm_joint_ids = resolve_joint_ids(robot.joint_names, RIGHT_ARM_JOINT_NAMES)
        self._arm_action_columns = [POLICY_JOINT_NAMES.index(name) for name in RIGHT_ARM_JOINT_NAMES]
        self._jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in self._arm_joint_ids]
        self._end_effector_body_id = robot.body_names.index(RIGHT_END_EFFECTOR_BODY_NAME)
        self._ik_cfg = ik_cfg or B0IkCfg(damping=0.03, maximum_joint_step_rad=0.01, maximum_target_step_rad=0.005)
        self._arm_target_rad: torch.Tensor | None = None

    @property
    def end_effector_body_id(self) -> int:
        """Return the right-wrist body index in the articulation."""
        return self._end_effector_body_id

    @property
    def arm_joint_ids(self) -> list[int]:
        """Return articulation indices for the controlled right-arm joints."""
        return self._arm_joint_ids

    @property
    def arm_target_rad(self) -> torch.Tensor:
        """Return the latest smooth arm position target [rad], shape ``[N, 4]``."""
        if self._arm_target_rad is None:
            raise RuntimeError("Call reset() before reading the B0 arm target.")
        return self._arm_target_rad

    def reset(self) -> None:
        """Initialize the commanded arm pose from the current articulation state."""
        self._arm_target_rad = self._robot.data.joint_pos.torch[:, self._arm_joint_ids].clone()

    def action(self, target_position_w_m: torch.Tensor) -> torch.Tensor:
        """Compute one smooth policy action for a world-frame wrist target.

        Args:
            target_position_w_m: Desired wrist position [m], shape ``[N, 3]``.

        Returns:
            Unitless policy actions, shape ``[N, 21]``.
        """
        if self._arm_target_rad is None:
            raise RuntimeError("Call reset() before generating B0 actions.")
        wrist_position_w_m = self._robot.data.body_pos_w.torch[:, self._end_effector_body_id]
        jacobian_w = self._robot.data.body_link_jacobian_w.torch[:, self._end_effector_body_id, :3, self._jacobian_joint_ids]
        arm_position_rad = self._robot.data.joint_pos.torch[:, self._arm_joint_ids]
        lower_limit_rad = self._robot.data.soft_joint_pos_limits.torch[:, self._arm_joint_ids, 0]
        upper_limit_rad = self._robot.data.soft_joint_pos_limits.torch[:, self._arm_joint_ids, 1]
        nominal_arm_rad = self._robot.data.default_joint_pos.torch[:, self._arm_joint_ids]
        desired_arm_target_rad = damped_least_squares_step(
            wrist_position_w_m, target_position_w_m, jacobian_w, arm_position_rad, lower_limit_rad, upper_limit_rad, self._ik_cfg
        )
        self._arm_target_rad = smooth_joint_position_target(
            self._arm_target_rad, desired_arm_target_rad, self._ik_cfg.maximum_target_step_rad
        )
        action = torch.zeros((wrist_position_w_m.shape[0], len(POLICY_JOINT_NAMES)), device=self._device)
        action[:, self._arm_action_columns] = ((self._arm_target_rad - nominal_arm_rad) / ACTION_SCALE_RAD).clamp(-1.0, 1.0)
        return action
