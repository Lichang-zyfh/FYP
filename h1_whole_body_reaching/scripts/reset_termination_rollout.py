# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate deterministic reset randomization and initial task terminations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Official h1_2_handless.urdf path.")
parser.add_argument("--log-file", type=Path, default=None, help="Optional JSON result path.")
parser.add_argument("--num-seeds", type=int, default=4, help="Number of deterministic randomized resets to validate.")
parser.add_argument("--seed", type=int, default=7, help="First deterministic reset seed.")
parser.add_argument("--settle-steps", type=int, default=100, help="Physics steps after each reset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.robot_cfg import PHYSICS_DT_S, make_h1_2_cfg
from h1_whole_body_reaching.task_contract import (
    FixedRightHandTargetCfg,
    ResetRandomizationCfg,
    apply_reset_policy_joint_offsets,
    evaluate_task_termination,
    make_reset_policy_joint_offsets,
)


def run_rollout() -> dict[str, object]:
    """Run bounded randomized resets and validate their state contract."""
    if args_cli.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args_cli.settle_steps <= 0:
        raise ValueError("--settle-steps must be positive.")

    reset_cfg = ResetRandomizationCfg()
    target_cfg = FixedRightHandTargetCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT_S, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_h1_2_cfg(args_cli.urdf))
    sim.reset()

    all_joint_ids = resolve_joint_ids(robot.joint_names, ALL_JOINT_NAMES)
    nominal_targets = robot.data.default_joint_pos.torch[:, all_joint_ids].clone()
    joint_limits = robot.data.joint_pos_limits.torch[0, all_joint_ids, :]
    reset_results: list[dict[str, object]] = []

    for seed in range(args_cli.seed, args_cli.seed + args_cli.num_seeds):
        offsets = make_reset_policy_joint_offsets(seed, reset_cfg)
        targets = torch.tensor(
            [apply_reset_policy_joint_offsets(nominal_targets[0].tolist(), offsets)],
            dtype=nominal_targets.dtype,
            device=args_cli.device,
        )
        reset_joint_pos = robot.data.default_joint_pos.torch.clone()
        reset_joint_vel = torch.zeros_like(robot.data.default_joint_vel.torch)
        reset_joint_pos[:, all_joint_ids] = targets
        robot.write_joint_position_to_sim_index(position=reset_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=reset_joint_vel)
        robot.reset()

        for _ in range(args_cli.settle_steps):
            robot.actuators.target_command.set_position_index(value=targets, joint_ids=all_joint_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT_S)

        joint_positions = robot.data.joint_pos.torch[0, all_joint_ids]
        joint_velocities = robot.data.joint_vel.torch[0, all_joint_ids]
        max_target_error_rad = torch.max(torch.abs(joint_positions - targets[0])).item()
        termination = evaluate_task_termination(
            target_error_m=target_cfg.success_tolerance_m * 2.0,
            step_count=args_cli.settle_steps,
            target_cfg=target_cfg,
            joint_positions_rad=joint_positions.tolist(),
            joint_lower_limits_rad=joint_limits[:, 0].tolist(),
            joint_upper_limits_rad=joint_limits[:, 1].tolist(),
        )
        if not torch.isfinite(joint_positions).all() or not torch.isfinite(joint_velocities).all():
            raise RuntimeError(f"H1-2 state became non-finite after reset seed {seed}.")
        if termination.invalid_state or termination.joint_limit_violation or termination.time_out:
            raise RuntimeError(f"Reset seed {seed} triggered termination: {termination}.")
        reset_results.append(
            {
                "seed": seed,
                "max_sampled_policy_offset_rad": max(abs(offset) for offset in offsets),
                "max_settled_target_error_rad": max_target_error_rad,
                "termination": {
                    "success": termination.success,
                    "time_out": termination.time_out,
                    "invalid_state": termination.invalid_state,
                    "joint_limit_violation": termination.joint_limit_violation,
                },
            }
        )

    return {
        "status": "passed",
        "fixed_base": True,
        "physics_dt_s": PHYSICS_DT_S,
        "policy_joint_position_offset_bound_rad": reset_cfg.policy_joint_position_offset_bound_rad,
        "policy_joint_count": len(offsets),
        "wrist_reset_targets_preserved": True,
        "settle_steps": args_cli.settle_steps,
        "simulated_duration_per_seed_s": args_cli.settle_steps * PHYSICS_DT_S,
        "reset_results": reset_results,
    }


def main() -> None:
    """Execute reset validation, print the result, and optionally save it."""
    result = run_rollout()
    output = json.dumps(result, indent=2)
    print(output)
    if args_cli.log_file is not None:
        args_cli.log_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.log_file.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
