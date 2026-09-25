# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Phase 2 fixed-base H1-2 reset and zero-action rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Official h1_2_handless.urdf path.")
parser.add_argument("--steps", type=int, default=200, help="Number of physics steps to simulate.")
parser.add_argument("--log-file", type=Path, default=None, help="Optional JSON result path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, POLICY_JOINT_NAMES, WRIST_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.phase_2_validation import forward_kinematics
from h1_whole_body_reaching.robot_cfg import ACTION_SCALE_RAD, CONTROL_DECIMATION, PHYSICS_DT_S, make_h1_2_cfg
from h1_whole_body_reaching.task_contract import RIGHT_END_EFFECTOR_BODY_NAME


def run_rollout() -> dict[str, object]:
    """Run reset plus zero action and return a machine-readable result."""
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT_S, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(make_h1_2_cfg(args_cli.urdf))
    sim.reset()

    actual_joint_names = robot.joint_names
    all_joint_ids = resolve_joint_ids(actual_joint_names, ALL_JOINT_NAMES)
    policy_joint_ids = resolve_joint_ids(actual_joint_names, POLICY_JOINT_NAMES)
    wrist_joint_ids = resolve_joint_ids(actual_joint_names, WRIST_JOINT_NAMES)
    end_effector_body_id = robot.body_names.index(RIGHT_END_EFFECTOR_BODY_NAME)
    nominal_targets = robot.data.default_joint_pos.torch[:, all_joint_ids].clone()

    robot.write_joint_position_to_sim_index(position=nominal_targets, joint_ids=all_joint_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(nominal_targets), joint_ids=all_joint_ids)
    robot.reset()

    max_position_error_rad = 0.0
    for _ in range(args_cli.steps):
        robot.actuators.target_command.set_position_index(value=nominal_targets, joint_ids=all_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT_S)
        joint_pos = robot.data.joint_pos.torch[:, all_joint_ids]
        joint_vel = robot.data.joint_vel.torch[:, all_joint_ids]
        if not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all():
            raise RuntimeError("H1-2 state became non-finite during zero-action rollout.")
        max_position_error_rad = max(max_position_error_rad, float(torch.max(torch.abs(joint_pos - nominal_targets))))

    joint_positions = robot.data.joint_pos.torch[0, all_joint_ids].tolist()
    joint_position_by_name = dict(zip(ALL_JOINT_NAMES, joint_positions, strict=True))
    root_position_w_m = np.asarray(robot.data.root_pos_w.torch[0].tolist())
    fk_end_effector_position_w_m = forward_kinematics(args_cli.urdf, joint_position_by_name, root_position_w_m)[
        RIGHT_END_EFFECTOR_BODY_NAME
    ]
    simulator_end_effector_position_w_m = np.asarray(robot.data.body_pos_w.torch[0, end_effector_body_id].tolist())
    end_effector_fk_error_m = float(np.linalg.norm(fk_end_effector_position_w_m - simulator_end_effector_position_w_m))
    if end_effector_fk_error_m > 2e-5:
        raise RuntimeError(f"URDF FK and Isaac Sim end-effector positions diverged by {end_effector_fk_error_m} m.")

    wrist_targets = nominal_targets[:, [all_joint_ids.index(index) for index in wrist_joint_ids]]
    result = {
        "status": "passed",
        "fixed_base": True,
        "physics_dt_s": PHYSICS_DT_S,
        "control_decimation": CONTROL_DECIMATION,
        "control_frequency_hz": 1.0 / (PHYSICS_DT_S * CONTROL_DECIMATION),
        "steps": args_cli.steps,
        "simulated_duration_s": args_cli.steps * PHYSICS_DT_S,
        "joint_count": len(actual_joint_names),
        "joint_names": actual_joint_names,
        "policy_joint_count": len(policy_joint_ids),
        "policy_joint_names": list(POLICY_JOINT_NAMES),
        "wrist_joint_count": len(wrist_joint_ids),
        "wrist_joint_names": list(WRIST_JOINT_NAMES),
        "wrist_nominal_targets_rad": wrist_targets[0].tolist(),
        "action_scale_rad": ACTION_SCALE_RAD,
        "max_position_error_rad": max_position_error_rad,
        "end_effector_body_name": RIGHT_END_EFFECTOR_BODY_NAME,
        "urdf_fk_to_isaac_sim_end_effector_error_m": end_effector_fk_error_m,
    }
    return result


def main() -> None:
    """Execute the rollout, print the result, and optionally save it."""
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
