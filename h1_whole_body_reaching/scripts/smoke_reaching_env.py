# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Instantiate and step the registered vectorized Phase 3 reaching environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Official h1_2_handless.urdf path.")
parser.add_argument("--num-envs", type=int, default=2, help="Number of vectorized environments.")
parser.add_argument("--steps", type=int, default=4, help="Number of random policy steps.")
parser.add_argument("--free-base", action="store_true", help="Use the free-base H1-2 articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.reaching_env_cfg import make_h1_reaching_env_cfg


def main() -> None:
    """Run reset and random-action steps, checking observable vector contracts."""
    cfg = make_h1_reaching_env_cfg(
        args_cli.urdf,
        args_cli.num_envs,
        args_cli.device,
        fixed_base=not args_cli.free_base,
    )
    env = gym.make("H1-2-Whole-Body-Reaching-Direct-v0", cfg=cfg)
    observations, _ = env.reset(seed=7)
    if observations["policy"].shape != (args_cli.num_envs, 66):
        raise RuntimeError(f"Unexpected observation shape: {observations['policy'].shape}")
    reward_min = float("inf")
    reward_max = float("-inf")
    terminated_count = 0
    truncated_count = 0
    for _ in range(args_cli.steps):
        actions = torch.zeros(args_cli.num_envs, 21, device=env.unwrapped.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(rewards).all():
            raise RuntimeError("Environment emitted non-finite observations or rewards.")
        if terminated.shape != (args_cli.num_envs,) or truncated.shape != (args_cli.num_envs,):
            raise RuntimeError("Termination buffers have an unexpected shape.")
        reward_min = min(reward_min, rewards.min().item())
        reward_max = max(reward_max, rewards.max().item())
        terminated_count += terminated.count_nonzero().item()
        truncated_count += truncated.count_nonzero().item()
    print({
        "status": "passed",
        "num_envs": args_cli.num_envs,
        "fixed_base": cfg.fixed_base,
        "steps": args_cli.steps,
        "observation_dimension": 66,
        "reward_range": [reward_min, reward_max],
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    })
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
