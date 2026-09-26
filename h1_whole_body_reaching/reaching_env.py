# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized H1-2 position-reaching environment for Phase 3."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, POLICY_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.reaching_env_cfg import H1ReachingEnvCfg
from h1_whole_body_reaching.robot_cfg import ACTION_SCALE_RAD
from h1_whole_body_reaching.standing_controller import StandingStabilizerCfg, apply_standing_stabilizer
from h1_whole_body_reaching.task_contract import (
    RIGHT_END_EFFECTOR_BODY_NAME,
    validate_reaching_task_cfg,
    validate_standing_task_cfg,
)


class H1ReachingEnv(DirectRLEnv):
    """Vectorized right-wrist target-reaching environment.

    Fixed-base instances retain the initial reaching-only termination behavior.
    Free-base instances additionally terminate on a low pelvis, insufficient
    uprightness, or simultaneous loss of both foot contacts.
    """

    cfg: H1ReachingEnvCfg

    def __init__(self, cfg: H1ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        validate_reaching_task_cfg(cfg.task)
        validate_standing_task_cfg(cfg.standing_task)
        super().__init__(cfg, render_mode, **kwargs)
        self._robot: Articulation = self.scene["robot"]
        self._feet_contact: ContactSensor = self.scene["feet_contact"]
        self._prohibited_torso_contact: ContactSensor = self.scene["prohibited_torso_contact"]
        self._all_joint_ids = resolve_joint_ids(self._robot.joint_names, ALL_JOINT_NAMES)
        self._all_joint_names = tuple(self._robot.joint_names[index] for index in self._all_joint_ids)
        self._policy_joint_ids = resolve_joint_ids(self._robot.joint_names, POLICY_JOINT_NAMES)
        self._end_effector_body_id = self._robot.body_names.index(RIGHT_END_EFFECTOR_BODY_NAME)
        self._pelvis_body_id = self._robot.body_names.index("pelvis")
        self._actions = torch.zeros(self.num_envs, len(POLICY_JOINT_NAMES), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._targets_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._success_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_hold_required = round(cfg.task.success_hold_duration_s / self.step_dt)
        self._contact_grace_steps = round(cfg.task.free_base_contact_grace_duration_s / self.step_dt)
        self._tilt_grace_steps = round(cfg.task.free_base_tilt_grace_duration_s / self.step_dt)
        self._tilt_violation_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)
        self._standing_root_reference_w = self._robot.data.default_root_state.torch[:, :3].clone()
        self._standing_root_reference_w += self.scene.env_origins
        self._standing_stabilizer_cfg = StandingStabilizerCfg(**self.cfg.free_base_stabilizer)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)
        defaults = self._robot.data.default_joint_pos.torch[:, self._all_joint_ids]
        self._joint_targets = defaults.clone()
        policy_columns = [self._all_joint_ids.index(joint_id) for joint_id in self._policy_joint_ids]
        self._joint_targets[:, policy_columns] += ACTION_SCALE_RAD * self._actions
        if not self.cfg.fixed_base and self.cfg.enable_standing_stabilizer:
            for joint_name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
                self._joint_targets[:, self._all_joint_names.index(joint_name)] += (
                    self.cfg.free_base_shoulder_pitch_offset_rad
                )

    def _apply_action(self) -> None:
        joint_targets = self._joint_targets
        if not self.cfg.fixed_base:
            ground_forces_w_n = self._feet_contact.data.normal_force_matrix_w.torch[:, :, 0, :]
            ground_contact_positions_w_m = self._feet_contact.data.contact_pos_w.torch[:, :, 0, :]
            normal_force_n = ground_forces_w_n[:, :, 2].clamp_min(0.0)
            total_normal_force_n = normal_force_n.sum(dim=1)
            support_center_w_m = (
                ground_contact_positions_w_m.nan_to_num() * normal_force_n.unsqueeze(-1)
            ).sum(dim=1)
            support_center_w_m /= total_normal_force_n.clamp_min(1.0).unsqueeze(-1)
            support_valid = torch.isfinite(ground_contact_positions_w_m).all(dim=(1, 2)) & (
                total_normal_force_n >= self.cfg.task.free_base_min_foot_contact_force_n
            )
            pelvis_up_axis_w = quat_apply(
                self._robot.data.root_quat_w.torch,
                torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, 3),
            )
            joint_targets = apply_standing_stabilizer(
                self._joint_targets,
                self._all_joint_names,
                pelvis_up_axis_w,
                self._robot.data.root_ang_vel_w.torch,
                self._robot.data.root_pos_w.torch,
                self._standing_root_reference_w,
                self._robot.data.root_lin_vel_w.torch,
                self._standing_stabilizer_cfg,
                support_center_w_m,
                support_valid,
            )
        self._robot.set_joint_position_target_index(target=joint_targets, joint_ids=self._all_joint_ids)
        if self.cfg.standing_push_force_n > 0.0:
            active = (self.episode_length_buf >= self.cfg.standing_push_start_step) & (
                self.episode_length_buf < self.cfg.standing_push_start_step + self.cfg.standing_push_duration_steps
            )
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            forces[:, 0, 0] = self.cfg.standing_push_force_n * active * torch.where(
                torch.arange(self.num_envs, device=self.device) % 2 == 0, 1.0, -1.0
            )
            self._robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), body_ids=[self._pelvis_body_id], is_global=True
            )

    def _target_error(self) -> torch.Tensor:
        hand_pos_w = self._robot.data.body_pos_w.torch[:, self._end_effector_body_id]
        return torch.linalg.vector_norm(self._targets_w - hand_pos_w, dim=-1)

    def _free_base_safety_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fall and both-feet-lost-contact masks for free-base instances."""
        if self.cfg.fixed_base:
            false_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return false_mask, false_mask

        root_position_w = self._robot.data.root_pos_w.torch
        root_quaternion_w = self._robot.data.root_quat_w.torch
        pelvis_up_axis_w = quat_apply(
            root_quaternion_w,
            torch.tensor((0.0, 0.0, 1.0), device=self.device).expand_as(root_position_w),
        )
        low_root = root_position_w[:, 2] < self.cfg.task.free_base_fall_height_m
        tilted = pelvis_up_axis_w[:, 2] < self.cfg.task.free_base_min_upright_cosine
        self._tilt_violation_steps = torch.where(tilted, self._tilt_violation_steps + 1, 0)
        fallen = low_root | (self._tilt_violation_steps > self._tilt_grace_steps)
        foot_force_norms = torch.linalg.vector_norm(self._feet_contact.data.net_normal_forces_w.torch, dim=-1)
        both_feet_lost_contact = torch.all(
            foot_force_norms < self.cfg.task.free_base_min_foot_contact_force_n, dim=-1
        )
        contact_grace_elapsed = self.episode_length_buf >= self._contact_grace_steps
        return fallen, both_feet_lost_contact & contact_grace_elapsed

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.standing_only:
            base_feature = quat_apply(
                self._robot.data.root_quat_w.torch,
                torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, 3),
            )
        else:
            hand_pos_w = self._robot.data.body_pos_w.torch[:, self._end_effector_body_id]
            base_feature = self._targets_w - hand_pos_w
        observation = torch.cat((
            self._robot.data.joint_pos.torch[:, self._policy_joint_ids] - self._robot.data.default_joint_pos.torch[:, self._policy_joint_ids],
            self._robot.data.joint_vel.torch[:, self._policy_joint_ids],
            base_feature,
            self._previous_actions,
        ), dim=-1)
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.standing_only:
            root_position_w = self._robot.data.root_pos_w.torch
            pelvis_up_axis_w = quat_apply(
                self._robot.data.root_quat_w.torch,
                torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, 3),
            )
            upright = pelvis_up_axis_w[:, 2].clamp(min=0.0, max=1.0)
            reference_height = self._standing_root_reference_w[:, 2]
            height_error = (root_position_w[:, 2] - reference_height).abs()
            height_reward = torch.exp(-height_error / self.cfg.standing_task.base_height_error_scale_m)
            foot_force_norms = torch.linalg.vector_norm(self._feet_contact.data.net_normal_forces_w.torch, dim=-1)
            bilateral_contact = torch.all(
                foot_force_norms >= self.cfg.task.free_base_min_foot_contact_force_n, dim=-1
            ).float()
            posture_error = self._robot.data.joint_pos.torch[:, self._policy_joint_ids] - self._robot.data.default_joint_pos.torch[:, self._policy_joint_ids]
            posture_reward = torch.exp(-posture_error.square().mean(dim=-1) / self.cfg.standing_task.posture_error_scale_rad**2)
            action_delta = self._actions - self._previous_actions
            action_smoothness = torch.exp(-action_delta.square().mean(dim=-1))
            fallen, lost_foot_contact = self._free_base_safety_masks()
            reward = (
                self.cfg.standing_task.upright_reward_weight * upright
                + self.cfg.standing_task.base_height_reward_weight * height_reward
                + self.cfg.standing_task.foot_contact_reward_weight * bilateral_contact
                + self.cfg.standing_task.posture_reward_weight * posture_reward
                + self.cfg.standing_task.action_smoothness_reward_weight * action_smoothness
                - self.cfg.standing_task.fall_penalty * (fallen | lost_foot_contact).float()
            )
            self._previous_actions = self._actions.clone()
            return reward
        target_error = self._target_error()
        held_success = self._success_hold_steps >= self._success_hold_required
        reward = self.cfg.task.reaching_reward_scale * (1.0 - target_error.clamp(max=1.0))
        reward += self.cfg.task.success_reward * held_success.float()
        self._episode_error_sum += target_error
        self._episode_steps += 1.0
        self._previous_actions = self._actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        error = self._target_error()
        if self.cfg.standing_only:
            held_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            within_tolerance = error <= self.cfg.task.success_tolerance_m
            self._success_hold_steps = torch.where(within_tolerance, self._success_hold_steps + 1, 0)
            held_success = self._success_hold_steps >= self._success_hold_required
        invalid = ~torch.isfinite(error) | ~torch.isfinite(
            self._robot.data.joint_pos.torch[:, self._all_joint_ids]
        ).all(dim=-1)
        fallen, lost_foot_contact = self._free_base_safety_masks()
        prohibited_torso_contact = torch.any(
            torch.linalg.vector_norm(self._prohibited_torso_contact.data.net_normal_forces_w.torch, dim=-1)
            >= self.cfg.task.free_base_min_foot_contact_force_n,
            dim=-1,
        )
        root_position_w = self._robot.data.root_pos_w.torch
        root_quaternion_w = self._robot.data.root_quat_w.torch
        root_up_cosine = 1.0 - 2.0 * (root_quaternion_w[:, 1].square() + root_quaternion_w[:, 2].square())
        self.extras["free_base_root_height_m"] = root_position_w[:, 2].clone()
        self.extras["free_base_root_up_cosine"] = root_up_cosine.clone()
        pelvis_up_axis_w = quat_apply(
            root_quaternion_w,
            torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, 3),
        )
        self.extras["free_base_up_axis_w"] = pelvis_up_axis_w.clone()
        self.extras["free_base_root_linear_velocity_w_mps"] = self._robot.data.root_lin_vel_w.torch.clone()
        self.extras["free_base_root_angular_velocity_w_radps"] = self._robot.data.root_ang_vel_w.torch.clone()
        self.extras["prohibited_torso_contact"] = prohibited_torso_contact.clone()
        terminated = invalid | fallen | lost_foot_contact | held_success
        causes = torch.stack((invalid, fallen, lost_foot_contact, held_success), dim=-1)
        self.extras["termination_causes"] = causes.clone()
        self.extras["task_terminated"] = terminated.clone()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task_time_out"] = time_out.clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.reset(env_ids)
        self._robot.permanent_wrench_composer.reset(env_ids=env_ids)
        self._feet_contact.reset(env_ids)
        self._prohibited_torso_contact.reset(env_ids)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.cfg.standing_only:
            self._targets_w[env_ids] = 0.0
        else:
            lower = torch.tensor(self.cfg.task.target_lower_w_m, device=self.device)
            upper = torch.tensor(self.cfg.task.target_upper_w_m, device=self.device)
            self._targets_w[env_ids] = (
                self.scene.env_origins[env_ids]
                + lower
                + (upper - lower) * torch.rand(count, 3, device=self.device)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._success_hold_steps[env_ids] = 0
        self._tilt_violation_steps[env_ids] = 0
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_offsets = torch.empty(count, len(self._policy_joint_ids), device=self.device).uniform_(
            -self.cfg.reset_joint_offset_bound_rad,
            self.cfg.reset_joint_offset_bound_rad,
        )
        joint_pos[:, self._policy_joint_ids] += joint_offsets
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self._robot.data.default_joint_vel.torch[env_ids]), env_ids=env_ids
        )
        if not self.cfg.fixed_base:
            root_pose = self._robot.data.default_root_state.torch[env_ids, :7].clone()
            root_pose[:, :3] += self.scene.env_origins[env_ids]
            self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            self._robot.write_root_velocity_to_sim_index(
                root_velocity=self._robot.data.default_root_state.torch[env_ids, 7:].clone(), env_ids=env_ids
            )
        self._episode_error_sum[env_ids] = 0.0
        self._episode_steps[env_ids] = 0.0
