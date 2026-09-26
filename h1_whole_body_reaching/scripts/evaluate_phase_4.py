# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Persist Phase 4 per-episode metrics from free-base Isaac Lab evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory for episodes.jsonl and summary.json.")
parser.add_argument("--steps", type=int, default=500, help="Maximum control steps per episode.")
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 17), help="Evaluation reset seeds.")
parser.add_argument("--episodes-per-seed", type=int, default=1, help="Fresh episodes evaluated for each seed.")
parser.add_argument("--push-force-n", type=float, default=0.0, help="Optional world-x pelvis push magnitude [N].")
parser.add_argument("--push-duration-steps", type=int, default=10, help="Finite push duration [control steps].")
parser.add_argument(
    "--controller",
    choices=("zero", "b0", "ppo_standing"),
    default="zero",
    help="Action source: diagnostic zero action, B0 IK/PD, or a Phase 6 PPO standing checkpoint.",
)
parser.add_argument("--checkpoint", type=Path, help="RSL-RL checkpoint required by --controller ppo_standing.")
parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Fixed B0 wrist target [m].")
parser.add_argument("--target-ramp-s", type=float, default=1.0, help="B0 target interpolation duration [s].")
parser.add_argument("--scenario-name", help="Optional persisted scenario label; defaults to push or --case.")
parser.add_argument(
    "--case",
    choices=("nominal", "held_success", "timeout", "fall", "lost_contact", "zmp_outside_support"),
    default="nominal",
    help="Deterministic terminal-state scenario. Pushes remain available in every case.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

import h1_whole_body_reaching  # noqa: F401
from h1_whole_body_reaching.b0_controller import B0ActionGenerator
from h1_whole_body_reaching.b0_ik import interpolate_position_target
from h1_whole_body_reaching.dynamic_balance import compute_center_of_mass, compute_ground_zmp, compute_zmp_support_margin
from h1_whole_body_reaching.interfaces import POLICY_JOINT_NAMES
from h1_whole_body_reaching.phase_4_metrics import Phase4EpisodeAccumulator, write_phase_4_report
from h1_whole_body_reaching.reaching_env_cfg import (
    make_b0_reaching_env_cfg,
    make_h1_reaching_env_cfg,
    make_h1_standing_env_cfg,
)


def _pitch_roll_from_quaternion(quaternion_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pelvis pitch and roll from scalar-first world quaternions [rad]."""
    w, x, y, z = quaternion_w.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return pitch, roll


def _termination_cause(extras: dict[str, torch.Tensor], terminated: bool, truncated: bool) -> str:
    """Map mutually observable termination flags into one report label."""
    if truncated:
        return "timeout"
    if not terminated:
        return "horizon_complete"
    invalid, fallen, lost_contact, held_success = extras["termination_causes"][0].tolist()
    if invalid:
        return "invalid_state"
    if fallen:
        return "fall"
    if lost_contact:
        return "lost_foot_contact"
    if held_success:
        return "held_reaching_success"
    return "terminated"


def main() -> None:
    """Run deterministic episodes and write Phase 4 metric records."""
    if args_cli.steps <= 0 or args_cli.episodes_per_seed <= 0 or args_cli.push_duration_steps <= 0 or args_cli.target_ramp_s <= 0.0:
        raise ValueError("--steps, --episodes-per-seed, --push-duration-steps, and --target-ramp-s must be positive.")
    if args_cli.target and args_cli.controller != "b0":
        raise ValueError("--target is only valid with --controller b0.")
    if args_cli.controller == "ppo_standing" and args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required with --controller ppo_standing.")
    if args_cli.checkpoint is not None and args_cli.controller != "ppo_standing":
        raise ValueError("--checkpoint is only valid with --controller ppo_standing.")
    if args_cli.controller == "b0":
        cfg = make_b0_reaching_env_cfg(args_cli.urdf, 1, args_cli.device)
    elif args_cli.controller == "ppo_standing":
        cfg = make_h1_standing_env_cfg(args_cli.urdf, 1, args_cli.device)
    else:
        cfg = make_h1_reaching_env_cfg(args_cli.urdf, 1, args_cli.device, fixed_base=False)
    env = gym.make("H1-2-Whole-Body-Reaching-Direct-v0", cfg=cfg)
    records = []
    try:
        robot = env.unwrapped._robot
        feet_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_0/Robot/.*_ankle_roll_link",
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=["/World/Ground/CollisionPlane"],
                track_contact_points=True,
                track_friction_forces=True,
            )
        )
        robot.permanent_wrench_composer.reset()
        env.unwrapped.sim.reset()
        policy = None
        policy_env = None
        if args_cli.controller == "ppo_standing":
            from rsl_rl.runners import OnPolicyRunner

            from importlib.metadata import version

            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
            from h1_whole_body_reaching.agents.rsl_rl_ppo_cfg import H1StandingPPORunnerCfg

            policy_env = RslRlVecEnvWrapper(env)
            policy_cfg = H1StandingPPORunnerCfg()
            # Evaluation uses one environment; the optimizer is unused, but
            # RSL-RL still validates this constructor-time partition value.
            policy_cfg.algorithm.num_mini_batches = 1
            policy_cfg = handle_deprecated_rsl_rl_cfg(policy_cfg, version("rsl-rl-lib"))
            runner = OnPolicyRunner(
                policy_env,
                policy_cfg.to_dict(),
                log_dir=None,
                device=str(env.unwrapped.device),
            )
            runner.load(str(args_cli.checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        action = torch.zeros(1, len(POLICY_JOINT_NAMES), device=env.unwrapped.device)
        push = torch.tensor((args_cli.push_force_n, 0.0, 0.0), device=env.unwrapped.device)
        scenario = args_cli.scenario_name or ("push" if args_cli.push_force_n else args_cli.case)
        pelvis_id = robot.body_names.index("pelvis")
        for seed in args_cli.seeds:
            for episode_index in range(args_cli.episodes_per_seed):
                episode_seed = seed + episode_index
                if policy_env is not None:
                    # The wrapper returns a TensorDict view of the environment's
                    # observation buffer. Release the old view before resetting
                    # that buffer, then request the fresh policy observation.
                    observations = None
                    env.reset(seed=episode_seed)
                    observations = policy_env.get_observations()
                else:
                    env.reset(seed=episode_seed)
                accumulator = Phase4EpisodeAccumulator(scenario, seed, episode_index, env.unwrapped.step_dt)
                robot.permanent_wrench_composer.reset()
                controller = None
                final_target_w_m = None
                initial_wrist_w_m = None
                reaching_hold_steps = 0
                held_reaching_success = False
                if args_cli.controller == "b0":
                    controller = B0ActionGenerator(robot, env.unwrapped.device)
                    controller.reset()
                    initial_wrist_w_m = robot.data.body_pos_w.torch[:, controller.end_effector_body_id].clone()
                    if args_cli.target:
                        final_target_w_m = torch.tensor([args_cli.target], device=env.unwrapped.device)
                    else:
                        final_target_w_m = env.unwrapped._targets_w.clone()
                    env.unwrapped._targets_w[:] = final_target_w_m
                    if args_cli.push_force_n:
                        # A push evaluation needs the full post-push recovery
                        # window. Keep the physical fall/contact terminations,
                        # but evaluate held reaching independently below.
                        env.unwrapped._success_hold_required = args_cli.steps + 1
                if args_cli.case == "held_success":
                    env.unwrapped._targets_w[:] = robot.data.body_pos_w.torch[:, env.unwrapped._end_effector_body_id]
                    env.unwrapped._success_hold_steps[:] = env.unwrapped._success_hold_required - 1
                elif args_cli.case == "timeout":
                    env.unwrapped._targets_w[:] = robot.data.body_pos_w.torch[:, env.unwrapped._end_effector_body_id] + 10.0
                    env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length - 1
                elif args_cli.case == "fall":
                    pose = robot.data.root_link_pose_w.torch.clone()
                    pose[:, 2] = cfg.task.free_base_fall_height_m - 0.10
                    robot.write_root_pose_to_sim_index(root_pose=pose)
                if args_cli.push_force_n:
                    robot.permanent_wrench_composer.set_forces_and_torques_index(
                        forces=push.reshape(1, 1, 3), torques=torch.zeros_like(push).reshape(1, 1, 3),
                        body_ids=[pelvis_id], is_global=True,
                    )
                terminated = truncated = False
                extras: dict[str, torch.Tensor] = {}
                for step_index in range(args_cli.steps):
                    if controller is not None:
                        desired_target_w_m = interpolate_position_target(
                            initial_wrist_w_m,
                            final_target_w_m,
                            min(step_index * env.unwrapped.step_dt / args_cli.target_ramp_s, 1.0),
                        )
                        action = controller.action(desired_target_w_m)
                        env.unwrapped._targets_w[:] = final_target_w_m
                    # Let the contact sensor collect one physically valid sample before
                    # deliberately removing both feet from the ground.
                    if args_cli.case == "lost_contact" and step_index == 1:
                        pose = robot.data.root_link_pose_w.torch.clone()
                        pose[:, 2] = 2.0
                        robot.write_root_pose_to_sim_index(root_pose=pose)
                        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_vel_w.torch))
                        env.unwrapped.episode_length_buf[:] = env.unwrapped._contact_grace_steps
                    if args_cli.push_force_n and step_index == args_cli.push_duration_steps:
                        robot.permanent_wrench_composer.reset()
                    if policy is not None:
                        with torch.inference_mode():
                            action = policy(observations)
                        observations, _, dones, extras = policy_env.step(action)
                        truncated = bool(extras["task_time_out"].item())
                        terminated = bool(dones.item()) and not truncated
                    else:
                        _, _, term, trunc, extras = env.step(action)
                        terminated, truncated = bool(term.item()), bool(trunc.item())
                    if args_cli.controller in {"b0", "ppo_standing"} and (terminated or truncated):
                        # DirectRLEnv has already reset this scene. The preceding
                        # record is the last physical B0 state; never append the
                        # reset pose as terminal evaluation data.
                        break
                    if args_cli.controller == "b0" and args_cli.push_force_n:
                        within_tolerance = bool((env.unwrapped._target_error() <= cfg.task.success_tolerance_m).item())
                        reaching_hold_steps = reaching_hold_steps + 1 if within_tolerance else 0
                        held_reaching_success |= reaching_hold_steps >= round(cfg.task.success_hold_duration_s / env.unwrapped.step_dt)
                    feet_contact.update(env.unwrapped.step_dt)
                    positions = feet_contact.data.contact_pos_w.torch[:, :, 0, :]
                    forces = feet_contact.data.normal_force_matrix_w.torch[:, :, 0, :] + feet_contact.data.friction_force_matrix_w.torch[:, :, 0, :]
                    zmp, zmp_valid, _, _ = compute_ground_zmp(positions, forces, minimum_normal_force_n=cfg.task.free_base_min_foot_contact_force_n)
                    normal_forces = forces[..., 2]
                    if args_cli.case == "zmp_outside_support":
                        # Deterministic metric-path injection: retain the measured
                        # support geometry, but place a valid ZMP beyond its edge.
                        zmp = zmp.clone()
                        zmp[:, 0] = positions[:, :, 0].amax(dim=1) + 0.50
                        zmp_valid = torch.ones_like(zmp_valid)
                    _, support_valid, zmp_outside = compute_zmp_support_margin(zmp, zmp_valid, positions, normal_forces, minimum_contact_force_n=cfg.task.free_base_min_foot_contact_force_n)
                    com = compute_center_of_mass(robot.data.body_com_pos_w.torch, robot.data.body_mass.torch)
                    com_ground = com.clone()
                    com_ground[:, 2] = 0.0
                    com_margin, _, _ = compute_zmp_support_margin(com_ground, zmp_valid, positions, normal_forces, minimum_contact_force_n=cfg.task.free_base_min_foot_contact_force_n)
                    weights = normal_forces.clamp_min(0.0)
                    support_center = (positions.nan_to_num() * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                    pitch, roll = _pitch_roll_from_quaternion(robot.data.root_quat_w.torch)
                    foot_safe = (normal_forces >= cfg.task.free_base_min_foot_contact_force_n) & support_valid.unsqueeze(-1)
                    accumulator.add_step(
                        end_effector_error_m=(0.0 if args_cli.controller == "ppo_standing" else float(env.unwrapped._target_error().item())),
                        torso_pitch_rad=float(pitch.item()), torso_roll_rad=float(roll.item()),
                        com_support_margin_m=float(com_margin.item()), zmp_valid=bool(zmp_valid.item()), zmp_outside_support=bool(zmp_outside.item()),
                        zmp_deviation_m=float(torch.linalg.vector_norm(zmp[:, :2] - support_center[:, :2], dim=-1).item()) if bool(zmp_valid.item()) else None,
                        action=action[0].tolist(), joint_velocity_radps=robot.data.joint_vel.torch[0, env.unwrapped._policy_joint_ids].tolist(),
                        applied_torque_nm=robot.data.applied_torque.torch[0, env.unwrapped._policy_joint_ids].tolist(),
                        fell=bool(extras["termination_causes"][0, 1].item()), left_foot_safe=bool(foot_safe[0, 0].item()), right_foot_safe=bool(foot_safe[0, 1].item()),
                        prohibited_torso_contact=bool(extras["prohibited_torso_contact"].item()), disturbance_active=bool(args_cli.push_force_n and step_index < args_cli.push_duration_steps),
                    )
                    if terminated or truncated:
                        break
                robot.permanent_wrench_composer.reset()
                records.append(accumulator.finalize(
                    held_reaching_success=held_reaching_success or bool(extras["termination_causes"][0, 3].item()),
                    termination_cause=_termination_cause(extras, terminated, truncated),
                ))
        write_phase_4_report(records, args_cli.output_dir, {
            "step_dt_s": env.unwrapped.step_dt, "steps": args_cli.steps, "seeds": args_cli.seeds,
            "episodes_per_seed": args_cli.episodes_per_seed, "push_force_n": args_cli.push_force_n,
            "push_duration_steps": args_cli.push_duration_steps, "controller": args_cli.controller,
            "checkpoint": str(args_cli.checkpoint) if args_cli.checkpoint else None,
            "task_mode": "standing" if args_cli.controller == "ppo_standing" else "reaching",
            "target_position_w_m": args_cli.target, "target_ramp_s": args_cli.target_ramp_s,
        })
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
