# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run reproducible longer-horizon free-base balance and safety scenarios."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
parser.add_argument("--steps", type=int, default=40, help="Control steps in each non-injected scenario.")
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 17), help="Deterministic reset seeds.")
parser.add_argument(
    "--push-forces", type=float, nargs="+", default=(0.0, 60.0, 120.0), help="World-x push magnitudes [N]."
)
parser.add_argument(
    "--push-duration-steps", type=int, default=10, help="Control steps for which each short push is applied."
)
parser.add_argument("--arm-ramp-steps", type=int, default=20, help="Control steps used to ramp to the arm target.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.dynamic_balance import compute_ground_zmp, compute_zmp_support_margin
from h1_whole_body_reaching.evaluation_statistics import summarize_scenario_records
from h1_whole_body_reaching.interfaces import POLICY_JOINT_NAMES
from h1_whole_body_reaching.reaching_env_cfg import make_h1_reaching_env_cfg


def _read_stability(env: gym.Env, contact: ContactSensor) -> tuple[bool, bool, float]:
    """Return ZMP validity, violation, and signed support margin for one step."""
    positions_w_m = contact.data.contact_pos_w.torch[:, :, 0, :]
    forces_w_n = (
        contact.data.normal_force_matrix_w.torch[:, :, 0, :]
        + contact.data.friction_force_matrix_w.torch[:, :, 0, :]
    )
    zmp_w_m, zmp_valid, _, _ = compute_ground_zmp(
        positions_w_m,
        forces_w_n,
        minimum_normal_force_n=env.unwrapped.cfg.task.free_base_min_foot_contact_force_n,
    )
    margin_m, support_valid, outside = compute_zmp_support_margin(
        zmp_w_m,
        zmp_valid,
        positions_w_m,
        forces_w_n[..., 2],
        minimum_contact_force_n=env.unwrapped.cfg.task.free_base_min_foot_contact_force_n,
    )
    return bool(support_valid.item()), bool(outside.item()), float(margin_m.item())


def _run_scenario(
    env: gym.Env,
    contact: ContactSensor,
    seed: int,
    action: torch.Tensor,
    steps: int,
    push: torch.Tensor | None,
    push_duration_steps: int = 0,
    action_ramp_steps: int = 0,
) -> dict[str, object]:
    """Run one seeded non-terminal scenario and aggregate stability observables."""
    env.unwrapped._robot.permanent_wrench_composer.reset()
    env.unwrapped.sim.reset()
    env.reset(seed=seed)
    robot = env.unwrapped._robot
    robot.permanent_wrench_composer.reset()
    if push is not None:
        pelvis_id = robot.body_names.index("pelvis")
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=push.reshape(1, 1, 3),
            torques=torch.zeros_like(push).reshape(1, 1, 3),
            body_ids=[pelvis_id],
            is_global=True,
        )
    margins: list[float] = []
    valid_steps = 0
    violations = 0
    executed_steps = 0
    terminated = False
    truncated = False
    termination_diagnostics: dict[str, float | bool] = {}
    for step_index in range(steps):
        if push is not None and step_index == push_duration_steps:
            robot.permanent_wrench_composer.reset()
        executed_steps += 1
        if action_ramp_steps > 0:
            action_fraction = min((step_index + 1) / action_ramp_steps, 1.0)
            step_action = action * action_fraction
        else:
            step_action = action
        _, _, term, trunc, extras = env.step(step_action)
        returned_terminated = term.clone()
        returned_truncated = trunc.clone()
        terminated = bool(returned_terminated.item())
        truncated = bool(returned_truncated.item())
        contact.update(env.unwrapped.step_dt)
        valid, outside, margin = _read_stability(env, contact)
        valid_steps += int(valid)
        violations += int(outside)
        if valid:
            margins.append(margin)
        if terminated or truncated:
            termination_diagnostics = {
                "invalid_state": bool(extras["termination_causes"][0, 0].item()),
                "fallen": bool(extras["termination_causes"][0, 1].item()),
                "lost_foot_contact": bool(extras["termination_causes"][0, 2].item()),
                "held_success": bool(extras["termination_causes"][0, 3].item()),
                "task_terminated": bool(extras["task_terminated"].item()),
                "task_time_out": bool(extras["task_time_out"].item()),
                "root_height_m": float(extras["free_base_root_height_m"].item()),
                "root_up_cosine": float(extras["free_base_root_up_cosine"].item()),
                "root_up_axis_w": extras["free_base_up_axis_w"][0].tolist(),
                "root_linear_velocity_w_mps": extras["free_base_root_linear_velocity_w_mps"][0].tolist(),
                "root_angular_velocity_w_radps": extras["free_base_root_angular_velocity_w_radps"][0].tolist(),
            }
            break
    robot.permanent_wrench_composer.reset()
    return {
        "seed": seed,
        "requested_steps": steps,
        "executed_steps": executed_steps,
        "terminated": terminated,
        "truncated": truncated,
        "zmp_valid_steps": valid_steps,
        "zmp_violation_ratio": violations / valid_steps if valid_steps else 0.0,
        "minimum_support_margin_m": min(margins) if margins else 0.0,
        "termination_diagnostics": termination_diagnostics,
    }


def _injected_outcomes(env: gym.Env, action: torch.Tensor) -> dict[str, bool]:
    """Verify explicit fall and both-feet-lost-contact termination outcomes."""
    robot = env.unwrapped._robot
    env.reset(seed=7)
    pose = robot.data.root_link_pose_w.torch.clone()
    pose[:, 2] = env.unwrapped.cfg.task.free_base_fall_height_m - 0.10
    robot.write_root_pose_to_sim_index(root_pose=pose)
    _, _, fall, fall_timeout, _ = env.step(action)
    env.reset(seed=7)
    pose = robot.data.root_link_pose_w.torch.clone()
    pose[:, 2] = 2.0
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_vel_w.torch))
    env.unwrapped.episode_length_buf[:] = env.unwrapped._contact_grace_steps
    _, _, contact, contact_timeout, _ = env.step(action)
    return {
        "fall_terminated": bool(fall.item()) and not bool(fall_timeout.item()),
        "lost_contact_terminated": bool(contact.item()) and not bool(contact_timeout.item()),
    }


def main() -> None:
    """Run the fixed-seed scenario suite and reject missing required outcomes."""
    if args_cli.steps < 40:
        raise ValueError("--steps must be at least 40 for a longer-horizon scenario suite.")
    if args_cli.push_duration_steps <= 0:
        raise ValueError("--push-duration-steps must be positive.")
    if args_cli.arm_ramp_steps <= 0:
        raise ValueError("--arm-ramp-steps must be positive.")
    cfg = make_h1_reaching_env_cfg(args_cli.urdf, 1, args_cli.device, fixed_base=False)
    # This is a balance probe, not a reaching episode. A one-micrometre success
    # tolerance prevents randomly sampled reaching targets from ending its horizon.
    cfg.task = replace(cfg.task, success_tolerance_m=1.0e-6)
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
        zero = torch.zeros(1, len(POLICY_JOINT_NAMES), device=env.unwrapped.device)
        arm = zero.clone()
        arm[:, POLICY_JOINT_NAMES.index("right_shoulder_pitch_joint")] = -0.5
        static_records = [_run_scenario(env, contact, seed, zero, args_cli.steps, None) for seed in args_cli.seeds]
        arm_records = [
            _run_scenario(
                env,
                contact,
                seed,
                arm,
                args_cli.steps,
                None,
                action_ramp_steps=args_cli.arm_ramp_steps,
            )
            for seed in args_cli.seeds
        ]
        push_records = {
            force_n: [
                _run_scenario(
                    env,
                    contact,
                    seed,
                    zero,
                    args_cli.steps,
                    torch.tensor((force_n, 0.0, 0.0), device=env.unwrapped.device),
                    args_cli.push_duration_steps,
                )
                for seed in args_cli.seeds
            ]
            for force_n in args_cli.push_forces
        }
        outcomes = _injected_outcomes(env, zero)
        failure_reasons: list[str] = []
        if not all(
            item["executed_steps"] == args_cli.steps
            and item["zmp_valid_steps"] == args_cli.steps
            and not item["terminated"]
            and not item["truncated"]
            for item in static_records + arm_records
        ):
            failure_reasons.append("Static or arm-motion scenario did not survive the full measured horizon.")
        if any(
            item["zmp_valid_steps"] < args_cli.steps // 2 or item["truncated"]
            for records in push_records.values()
            for item in records
        ):
            failure_reasons.append("Push scenario did not provide enough valid non-timeout measurements.")
        if not all(outcomes.values()):
            failure_reasons.append("Required injected free-base termination outcome was not observed.")
        report = {
            "status": "passed" if not failure_reasons else "failed",
            "steps": args_cli.steps,
            "seeds": args_cli.seeds,
            "push_duration_steps": args_cli.push_duration_steps,
            "push_duration_s": args_cli.push_duration_steps * env.unwrapped.step_dt,
            "arm_ramp_steps": args_cli.arm_ramp_steps,
            "arm_ramp_duration_s": args_cli.arm_ramp_steps * env.unwrapped.step_dt,
            "success_tolerance_m": env.unwrapped.cfg.task.success_tolerance_m,
            "static": {"records": static_records, "summary": summarize_scenario_records(static_records)},
            "arm_motion": {"records": arm_records, "summary": summarize_scenario_records(arm_records)},
            "pushes": {
                force_n: {"records": records, "summary": summarize_scenario_records(records)}
                for force_n, records in push_records.items()
            },
            "injected_outcomes": outcomes,
            "failure_reasons": failure_reasons,
        }
        print(report, flush=True)
        if failure_reasons:
            raise RuntimeError("; ".join(failure_reasons))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
