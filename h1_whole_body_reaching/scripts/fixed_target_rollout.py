# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Phase 3 fixed right-hand target contract with a scripted action."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Official h1_2_handless.urdf path.")
parser.add_argument("--log-file", type=Path, default=None, help="Optional JSON result path.")
parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Optional right-wrist target [m].")
parser.add_argument(
    "--scripted-action", type=float, nargs=21, metavar="ACTION", help="Optional normalized action in frozen policy order."
)
parser.add_argument("--success-tolerance", type=float, default=None, help="Optional target success tolerance [m].")
parser.add_argument("--allow-miss", action="store_true", help="Record a target miss instead of raising an error.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg

from h1_whole_body_reaching.interfaces import (
    ALL_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    expand_policy_action,
    resolve_joint_ids,
)
from h1_whole_body_reaching.robot_cfg import ACTION_SCALE_RAD, PHYSICS_DT_S, make_h1_2_cfg
from h1_whole_body_reaching.task_contract import (
    RIGHT_END_EFFECTOR_BODY_NAME,
    FixedRightHandTargetCfg,
    evaluate_target_termination,
    make_scripted_target_action,
    make_target_observation,
    target_error_norm_m,
)


def _spawn_target_marker(target_position_w_m: tuple[float, float, float], device: str):
    """Spawn the fixed reaching target marker."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FixedRightHandTarget",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            )
        },
    )
    marker = marker_cfg.class_type(marker_cfg)
    marker.visualize(torch.tensor([target_position_w_m], device=device))
    return marker


def run_rollout() -> dict[str, object]:
    """Settle H1-2, execute a bounded target action, and validate termination."""
    target_cfg = FixedRightHandTargetCfg()
    if args_cli.target is not None:
        target_cfg = replace(target_cfg, position_w_m=tuple(args_cli.target))
    if args_cli.success_tolerance is not None:
        if args_cli.success_tolerance <= 0.0:
            raise ValueError("--success-tolerance must be positive.")
        target_cfg = replace(target_cfg, success_tolerance_m=args_cli.success_tolerance)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT_S, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    _spawn_target_marker(target_cfg.position_w_m, args_cli.device)
    robot = Articulation(make_h1_2_cfg(args_cli.urdf))
    sim.reset()

    all_joint_ids = resolve_joint_ids(robot.joint_names, ALL_JOINT_NAMES)
    end_effector_body_id = robot.body_names.index(RIGHT_END_EFFECTOR_BODY_NAME)
    nominal_targets = robot.data.default_joint_pos.torch[:, all_joint_ids].clone()

    def step_with_targets(targets: torch.Tensor, steps: int) -> None:
        for _ in range(steps):
            robot.actuators.target_command.set_position_index(value=targets, joint_ids=all_joint_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT_S)

    step_with_targets(nominal_targets, target_cfg.episode_length_steps)
    initial_hand_position = tuple(robot.data.body_pos_w.torch[0, end_effector_body_id].tolist())
    initial_observation = make_target_observation(initial_hand_position, target_cfg)
    initial_error_m = target_error_norm_m(initial_observation)

    scripted_action = args_cli.scripted_action or make_scripted_target_action()
    scripted_targets = torch.tensor(
        [expand_policy_action(scripted_action, nominal_targets[0].tolist(), ACTION_SCALE_RAD)],
        device=args_cli.device,
        dtype=nominal_targets.dtype,
    )
    step_with_targets(scripted_targets, target_cfg.episode_length_steps)
    final_hand_position = tuple(robot.data.body_pos_w.torch[0, end_effector_body_id].tolist())
    final_observation = make_target_observation(final_hand_position, target_cfg)
    final_error_m = target_error_norm_m(final_observation)
    termination = evaluate_target_termination(final_error_m, target_cfg.episode_length_steps, target_cfg)

    joint_pos = robot.data.joint_pos.torch[:, all_joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, all_joint_ids]
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all():
        raise RuntimeError("H1-2 state became non-finite during fixed-target rollout.")
    initially_within_tolerance = initial_error_m <= target_cfg.success_tolerance_m
    if initially_within_tolerance and not args_cli.allow_miss:
        raise RuntimeError("Fixed target is already satisfied before the scripted reaching action.")
    if not termination.success and not args_cli.allow_miss:
        raise RuntimeError(f"Scripted action missed fixed target by {final_error_m:.6f} m.")

    return {
        "status": "passed" if termination.success else "missed_target",
        "fixed_base": True,
        "physics_dt_s": PHYSICS_DT_S,
        "policy_joint_count": len(POLICY_JOINT_NAMES),
        "action_scale_rad": ACTION_SCALE_RAD,
        "end_effector_body_name": RIGHT_END_EFFECTOR_BODY_NAME,
        "target_position_w_m": list(target_cfg.position_w_m),
        "success_tolerance_m": target_cfg.success_tolerance_m,
        "observation": {"fields": ["right_wrist_position_w_m", "target_error_w_m"], "dimension": 6},
        "scripted_action": scripted_action,
        "initial_hand_position_w_m": list(initial_hand_position),
        "initial_target_error_m": initial_error_m,
        "initially_within_tolerance": initially_within_tolerance,
        "final_hand_position_w_m": list(final_hand_position),
        "final_target_error_m": final_error_m,
        "termination": {"success": termination.success, "time_out": termination.time_out},
        "steps": target_cfg.episode_length_steps,
        "simulated_duration_s": target_cfg.episode_length_steps * PHYSICS_DT_S,
    }


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
