# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate Phase 3 free-base safety and task termination paths on Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.reaching_env_cfg import make_h1_reaching_env_cfg


def _zero_action(env: gym.Env) -> torch.Tensor:
    """Return one zero normalized policy action batch."""
    return torch.zeros(env.unwrapped.num_envs, 21, device=env.unwrapped.device)


def _set_root_pose(env: gym.Env, height_m: float) -> None:
    """Set a stationary upright root pose for the sole free-base environment."""
    robot = env.unwrapped._robot
    root_pose = robot.data.root_link_pose_w.torch.clone()
    root_pose[:, 2] = height_m
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_vel_w.torch))


def _single_step(env: gym.Env) -> tuple[bool, bool]:
    """Step once and return scalar termination and timeout outputs."""
    _, _, terminated, truncated, _ = env.step(_zero_action(env))
    return bool(terminated.item()), bool(truncated.item())


def main() -> None:
    """Exercise normal, fall, lost-contact, timeout, and held-success paths."""
    cfg = make_h1_reaching_env_cfg(args_cli.urdf, 1, args_cli.device, fixed_base=False)
    env = gym.make("H1-2-Whole-Body-Reaching-Direct-v0", cfg=cfg)
    try:
        env.reset(seed=7)
        normal_terminated, normal_timeout = _single_step(env)
        if normal_terminated or normal_timeout:
            raise RuntimeError("A normal reset terminated before the contact grace period elapsed.")

        env.reset(seed=7)
        _set_root_pose(env, cfg.task.free_base_fall_height_m - 0.10)
        fall_terminated, fall_timeout = _single_step(env)
        if not fall_terminated or fall_timeout:
            raise RuntimeError("Low-root fall injection did not produce termination.")

        env.reset(seed=7)
        _set_root_pose(env, 2.0)
        env.unwrapped.episode_length_buf[:] = env.unwrapped._contact_grace_steps
        lost_contact_terminated, lost_contact_timeout = _single_step(env)
        if not lost_contact_terminated or lost_contact_timeout:
            raise RuntimeError("Both-feet-lost-contact injection did not produce termination.")

        env.reset(seed=7)
        hand_position_w = env.unwrapped._robot.data.body_pos_w.torch[:, env.unwrapped._end_effector_body_id].clone()
        env.unwrapped._targets_w[:] = hand_position_w + torch.tensor((1.0, 0.0, 0.0), device=env.unwrapped.device)
        env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length - 1
        timeout_terminated, timeout = _single_step(env)
        if timeout_terminated or not timeout:
            raise RuntimeError("Episode-length injection did not produce a pure timeout.")

        env.reset(seed=7)
        env.unwrapped._targets_w[:] = env.unwrapped._robot.data.body_pos_w.torch[:, env.unwrapped._end_effector_body_id]
        env.unwrapped._success_hold_steps[:] = env.unwrapped._success_hold_required - 1
        success_terminated, success_timeout = _single_step(env)
        if not success_terminated or success_timeout:
            raise RuntimeError("Held-success injection did not produce a pure termination.")

        print(
            {
                "status": "passed",
                "fixed_base": False,
                "normal_reset": {"terminated": normal_terminated, "timeout": normal_timeout},
                "fall": {"terminated": fall_terminated, "timeout": fall_timeout},
                "lost_foot_contact": {"terminated": lost_contact_terminated, "timeout": lost_contact_timeout},
                "timeout": {"terminated": timeout_terminated, "timeout": timeout},
                "held_success": {"terminated": success_terminated, "timeout": success_timeout},
                "thresholds": {
                    "fall_height_m": cfg.task.free_base_fall_height_m,
                    "min_upright_cosine": cfg.task.free_base_min_upright_cosine,
                    "min_foot_contact_force_n": cfg.task.free_base_min_foot_contact_force_n,
                    "contact_grace_duration_s": cfg.task.free_base_contact_grace_duration_s,
                },
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
