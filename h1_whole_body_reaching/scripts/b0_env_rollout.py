# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run B0 IK through the validated free-base DirectRLEnv execution path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
parser.add_argument("--output-file", type=Path, required=True, help="JSON rollout result path.")
parser.add_argument("--steps", type=int, default=100, help="Maximum 50 Hz control updates.")
parser.add_argument("--seed", type=int, default=7, help="Deterministic environment reset seed.")
parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Right-wrist target [m].")
parser.add_argument("--target-ramp-s", type=float, default=1.0, help="Target interpolation duration [s].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.b0_controller import B0ActionGenerator
from h1_whole_body_reaching.b0_ik import interpolate_position_target
from h1_whole_body_reaching.reaching_env_cfg import make_b0_reaching_env_cfg


def main() -> None:
    """Evaluate B0 with the environment's production stepping and standing path."""
    if args_cli.steps <= 0 or args_cli.target_ramp_s <= 0.0:
        raise ValueError("--steps and --target-ramp-s must be positive.")
    cfg = make_b0_reaching_env_cfg(args_cli.urdf, 1, args_cli.device)
    env = gym.make("H1-2-Whole-Body-Reaching-Direct-v0", cfg=cfg)
    try:
        env.reset(seed=args_cli.seed)
        robot = env.unwrapped._robot
        controller = B0ActionGenerator(robot, env.unwrapped.device)
        controller.reset()
        initial_wrist_w_m = robot.data.body_pos_w.torch[:, controller.end_effector_body_id].clone()
        initial_root_w_m = robot.data.root_pos_w.torch.clone()
        if args_cli.target:
            final_target_w_m = torch.tensor([args_cli.target], device=env.unwrapped.device)
        else:
            final_target_w_m = torch.tensor(
                [[(low + high) / 2.0 for low, high in zip(cfg.task.target_lower_w_m, cfg.task.target_upper_w_m, strict=True)]],
                device=env.unwrapped.device,
            )
        errors_m: list[float] = []
        terminated = False
        held_reaching_success = False
        last_wrist_w_m = initial_wrist_w_m.clone()
        last_root_w_m = initial_root_w_m.clone()
        last_arm_position_rad = robot.data.joint_pos.torch[:, controller.arm_joint_ids].clone()
        extras: dict[str, torch.Tensor] = {}
        for step in range(args_cli.steps):
            desired_target_w_m = interpolate_position_target(initial_wrist_w_m, final_target_w_m, min(step * env.unwrapped.step_dt / args_cli.target_ramp_s, 1.0))
            action = controller.action(desired_target_w_m)
            env.unwrapped._targets_w[:] = final_target_w_m
            last_wrist_w_m = robot.data.body_pos_w.torch[:, controller.end_effector_body_id].clone()
            last_root_w_m = robot.data.root_pos_w.torch.clone()
            last_arm_position_rad = robot.data.joint_pos.torch[:, controller.arm_joint_ids].clone()
            error_before_step_m = float(torch.linalg.vector_norm(final_target_w_m - last_wrist_w_m, dim=-1).item())
            _, _, term, trunc, extras = env.step(action)
            terminated = bool(term.item() or trunc.item())
            held_reaching_success = bool(extras["termination_causes"][0, 3].item())
            if terminated:
                # DirectRLEnv resets completed environments before returning.
                # The final simulator state is therefore unavailable here, but
                # a success termination proves its post-step error was in-bounds.
                errors_m.append(error_before_step_m)
                break
            error_m = float(torch.linalg.vector_norm(final_target_w_m - robot.data.body_pos_w.torch[:, controller.end_effector_body_id], dim=-1).item())
            errors_m.append(error_m)
        result = {
            "status": "passed" if held_reaching_success else "failed",
            "execution_path": "H1-2-Whole-Body-Reaching-Direct-v0 free-base",
            "seed": args_cli.seed,
            "final_end_effector_error_m": errors_m[-1],
            "mean_end_effector_error_m": sum(errors_m) / len(errors_m),
            "held_reaching_success": held_reaching_success,
            "termination_causes": extras["termination_causes"][0].tolist(),
            "control_steps": len(errors_m),
            "target_position_w_m": final_target_w_m[0].tolist(),
            "initial_end_effector_position_w_m": initial_wrist_w_m[0].tolist(),
            "final_end_effector_position_w_m": last_wrist_w_m[0].tolist(),
            "initial_root_position_w_m": initial_root_w_m[0].tolist(),
            "final_root_position_w_m": last_root_w_m[0].tolist(),
            "final_arm_joint_position_rad": last_arm_position_rad[0].tolist(),
            "final_arm_joint_target_rad": controller.arm_target_rad[0].tolist(),
        }
        args_cli.output_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
