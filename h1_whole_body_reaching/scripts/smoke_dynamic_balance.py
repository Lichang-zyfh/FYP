# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate free-base CoM and filtered-contact ZMP diagnostics on Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
parser.add_argument("--steps", type=int, default=8, help="Control steps in each contact-valid scenario.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.dynamic_balance import compute_center_of_mass, compute_ground_zmp, compute_zmp_support_margin
from h1_whole_body_reaching.interfaces import POLICY_JOINT_NAMES
from h1_whole_body_reaching.reaching_env_cfg import make_h1_reaching_env_cfg


def _measurement(
    env: gym.Env, contact: ContactSensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read CoM, ZMP, support margin, and filtered contact-wrench diagnostics."""
    robot = env.unwrapped._robot
    contact_positions_w_m = contact.data.contact_pos_w.torch[:, :, 0, :]
    contact_forces_w_n = (
        contact.data.normal_force_matrix_w.torch[:, :, 0, :]
        + contact.data.friction_force_matrix_w.torch[:, :, 0, :]
    )
    center_of_mass_w_m = compute_center_of_mass(robot.data.body_com_pos_w.torch, robot.data.body_mass.torch)
    zmp_w_m, valid, net_force_w_n, net_moment_w_nm = compute_ground_zmp(
        contact_positions_w_m,
        contact_forces_w_n,
        minimum_normal_force_n=env.unwrapped.cfg.task.free_base_min_foot_contact_force_n,
    )
    support_margin_m, support_valid, outside_support = compute_zmp_support_margin(
        zmp_w_m,
        valid,
        contact_positions_w_m,
        contact_forces_w_n[..., 2],
        minimum_contact_force_n=env.unwrapped.cfg.task.free_base_min_foot_contact_force_n,
    )
    return center_of_mass_w_m, zmp_w_m, valid, net_force_w_n, net_moment_w_nm, support_margin_m, support_valid, outside_support


def _collect(
    env: gym.Env, contact: ContactSensor, actions: torch.Tensor, steps: int, push_force_w_n: torch.Tensor | None = None
) -> list[tuple[torch.Tensor, ...]]:
    """Run a contact-valid scenario and retain diagnostic samples."""
    robot = env.unwrapped._robot
    robot.permanent_wrench_composer.reset()
    if push_force_w_n is not None:
        pelvis_body_id = robot.body_names.index("pelvis")
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=push_force_w_n.reshape(1, 1, 3),
            torques=torch.zeros_like(push_force_w_n).reshape(1, 1, 3),
            body_ids=[pelvis_body_id],
            is_global=True,
        )
    samples = []
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated.item() or truncated.item():
            raise RuntimeError("Contact-valid diagnostic scenario ended before measurement.")
        contact.update(env.unwrapped.step_dt)
        sample = _measurement(env, contact)
        if sample[2].item():
            samples.append(tuple(value.clone() for value in sample))
    robot.permanent_wrench_composer.reset()
    return samples


def _mean_position(samples: list[tuple[torch.Tensor, ...]], index: int) -> torch.Tensor:
    """Return the scalar-environment mean of one positional diagnostic field."""
    return torch.cat([sample[index] for sample in samples], dim=0).mean(dim=0)


def main() -> None:
    """Exercise static, arm-motion, push, and lost-contact diagnostic scenarios."""
    if args_cli.steps < 4:
        raise ValueError("--steps must be at least 4.")
    cfg = make_h1_reaching_env_cfg(args_cli.urdf, 1, args_cli.device, fixed_base=False)
    env = gym.make("H1-2-Whole-Body-Reaching-Direct-v0", cfg=cfg)
    try:
        contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_0/Robot/.*_ankle_roll_link",
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=["/World/Ground/CollisionPlane"],
                track_contact_points=True,
                track_friction_forces=True,
            )
        )
        env.unwrapped.sim.reset()
        zero_action = torch.zeros(1, 21, device=env.unwrapped.device)
        env.reset(seed=7)
        static_samples = _collect(env, contact, zero_action, args_cli.steps)
        if len(static_samples) < args_cli.steps // 2:
            raise RuntimeError("Static standing produced too few valid ZMP samples.")
        static_com_w_m = _mean_position(static_samples, 0)
        static_zmp_w_m = _mean_position(static_samples, 1)
        static_force_w_n = _mean_position(static_samples, 3)
        static_moment_w_nm = _mean_position(static_samples, 4)
        static_residuals_w_nm = torch.cat(
            [sample[4] - torch.linalg.cross(sample[1], sample[3], dim=-1) for sample in static_samples], dim=0
        )
        static_residual_w_nm = static_residuals_w_nm.mean(dim=0)
        if not torch.isfinite(static_residuals_w_nm).all() or torch.linalg.vector_norm(static_residuals_w_nm[:, :2], dim=-1).max() > 1.0e-3:
            raise RuntimeError("Static ZMP did not cancel the horizontal ground-contact moment.")
        if any(sample[7].item() or not sample[6].item() for sample in static_samples):
            raise RuntimeError("Static ZMP left the conservative support polygon.")

        env.reset(seed=7)
        arm_action = zero_action.clone()
        arm_action[:, POLICY_JOINT_NAMES.index("right_shoulder_pitch_joint")] = -0.5
        arm_samples = _collect(env, contact, arm_action, args_cli.steps)
        if len(arm_samples) < args_cli.steps // 2:
            raise RuntimeError("Arm-motion scenario produced too few valid ZMP samples.")
        if any(sample[7].item() or not sample[6].item() for sample in arm_samples):
            raise RuntimeError("Arm-motion ZMP left the conservative support polygon.")

        env.reset(seed=7)
        push_force_w_n = torch.tensor((120.0, 0.0, 0.0), device=env.unwrapped.device)
        push_samples = _collect(env, contact, zero_action, args_cli.steps, push_force_w_n)
        if len(push_samples) < args_cli.steps // 2:
            raise RuntimeError("Push scenario produced too few valid ZMP samples.")
        if any(sample[7].item() or not sample[6].item() for sample in push_samples):
            raise RuntimeError("Push ZMP left the conservative support polygon.")
        push_zmp_w_m = _mean_position(push_samples, 1)
        if torch.linalg.vector_norm((push_zmp_w_m - static_zmp_w_m)[:2]) < 1.0e-4:
            raise RuntimeError("Controlled push did not create a measurable ZMP displacement.")

        env.reset(seed=7)
        robot = env.unwrapped._robot
        root_pose = robot.data.root_link_pose_w.torch.clone()
        root_pose[:, 2] = 2.0
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_com_vel_w.torch))
        env.unwrapped.episode_length_buf[:] = env.unwrapped._contact_grace_steps
        _, _, lost_contact_terminated, _, _ = env.step(zero_action)
        contact.update(env.unwrapped.step_dt)
        _, lost_contact_zmp_w_m, lost_contact_valid, _, _, _, _, _ = _measurement(env, contact)
        if not lost_contact_terminated.item() or lost_contact_valid.item() or not torch.isfinite(lost_contact_zmp_w_m).all():
            raise RuntimeError("Lost-contact scenario did not return a safe invalid ZMP result.")

        print(
            {
                "status": "passed",
                "fixed_base": False,
                "valid_samples": {
                    "static": len(static_samples),
                    "arm_motion": len(arm_samples),
                    "controlled_push": len(push_samples),
                },
                "static": {
                    "center_of_mass_w_m": static_com_w_m.tolist(),
                    "zmp_w_m": static_zmp_w_m.tolist(),
                    "horizontal_moment_residual_nm": static_residual_w_nm[:2].tolist(),
                    "support_margin_m": float(_mean_position(static_samples, 5).item()),
                },
                "arm_motion": {"zmp_w_m": _mean_position(arm_samples, 1).tolist()},
                "controlled_push": {
                    "zmp_w_m": push_zmp_w_m.tolist(),
                    "zmp_displacement_xy_m": float(torch.linalg.vector_norm((push_zmp_w_m - static_zmp_w_m)[:2]).item()),
                    "support_margin_m": float(_mean_position(push_samples, 5).item()),
                },
                "lost_contact": {
                    "terminated": bool(lost_contact_terminated.item()),
                    "zmp_valid": bool(lost_contact_valid.item()),
                    "zmp_w_m": lost_contact_zmp_w_m.tolist(),
                },
                "support_polygon_contract": "parallel-foot common-longitudinal-overlap rectangle; half length 0.130 m, half width 0.043 m",
                "source_contract": "filtered PhysX ankle-roll ground contact force plus average contact point",
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
