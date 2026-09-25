# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the derived H1-2 asset through a free-base standing and contact smoke test."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Derived free-base H1-2 URDF path.")
parser.add_argument("--steps", type=int, default=500, help="Number of physics steps to simulate.")
parser.add_argument("--log-file", type=Path, default=None, help="Optional JSON result path.")
parser.add_argument("--trace-file", type=Path, default=None, help="Optional per-step controller trace JSON path.")
parser.add_argument(
    "--controller", choices=("nominal", "ankle", "sagittal", "whole_body"), default="nominal", help="Standing baseline controller."
)
parser.add_argument("--ankle-pitch-kp", type=float, default=1.0, help="Ankle pitch up-axis gain [rad/rad].")
parser.add_argument("--ankle-pitch-kd", type=float, default=0.15, help="Ankle pitch angular-velocity gain [s].")
parser.add_argument("--hip-pitch-ratio", type=float, default=0.0, help="Hip-pitch correction to ankle-correction ratio.")
parser.add_argument("--knee-ratio", type=float, default=0.0, help="Knee correction to ankle-correction ratio.")
parser.add_argument("--root-position-kp", type=float, default=0.0, help="Forward root-position feedback gain [rad/m].")
parser.add_argument("--root-velocity-kd", type=float, default=0.0, help="Forward root-velocity feedback gain [s/m].")
parser.add_argument("--support-position-kp", type=float, default=0.0, help="Pelvis-to-support-center feedback gain [rad/m].")
parser.add_argument("--capture-point-kp", type=float, default=0.0, help="Capture-point-to-support feedback gain [rad/m].")
parser.add_argument("--shoulder-pitch-ratio", type=float, default=0.0, help="Shoulder-pitch correction to ankle-correction ratio.")
parser.add_argument("--support-min-normal-force", type=float, default=20.0, help="Minimum total ground normal force for support feedback [N].")
parser.add_argument("--ankle-target-limit", type=float, default=0.20, help="Maximum ankle correction [rad].")
parser.add_argument("--ankle-roll-kp", type=float, default=0.0, help="Ankle roll up-axis gain [rad/rad].")
parser.add_argument("--ankle-roll-kd", type=float, default=0.0, help="Ankle roll angular-velocity gain [s].")
parser.add_argument("--ankle-roll-target-limit", type=float, default=0.10, help="Maximum ankle roll correction [rad].")
parser.add_argument("--left-hip-roll-ratio", type=float, default=0.0, help="Left hip-roll correction to ankle-roll correction ratio.")
parser.add_argument("--right-hip-roll-ratio", type=float, default=0.0, help="Right hip-roll correction to ankle-roll correction ratio.")
parser.add_argument(
    "--lateral-support-position-kp", type=float, default=0.0, help="Lateral pelvis-to-support-center feedback gain [rad/m]."
)
parser.add_argument(
    "--lateral-capture-point-kp", type=float, default=0.0, help="Lateral capture-point-to-support feedback gain [rad/m]."
)
parser.add_argument("--leg-damping", type=float, default=5.0, help="Leg implicit-PD damping [N m s/rad].")
parser.add_argument("--foot-stiffness", type=float, default=200.0, help="Foot implicit-PD stiffness [N m/rad].")
parser.add_argument("--foot-damping", type=float, default=10.0, help="Foot implicit-PD damping [N m s/rad].")
parser.add_argument("--shoulder-pitch-offset", type=float, default=0.0, help="Symmetric shoulder-pitch diagnostic offset [rad].")
parser.add_argument("--left-shoulder-roll-offset", type=float, default=0.0, help="Left shoulder-roll diagnostic offset [rad].")
parser.add_argument("--right-shoulder-roll-offset", type=float, default=0.0, help="Right shoulder-roll diagnostic offset [rad].")
parser.add_argument("--static-friction", type=float, default=1.0, help="Ground and robot static friction coefficient.")
parser.add_argument("--dynamic-friction", type=float, default=1.0, help="Ground and robot dynamic friction coefficient.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.robot_cfg import PHYSICS_DT_S, make_h1_2_cfg
from h1_whole_body_reaching.standing_controller import StandingStabilizerCfg, apply_standing_stabilizer


FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
GROUND_COLLISION_PRIM_PATH = "/World/Ground/CollisionPlane"


def _as_torch(value):
    """Return the torch view of an Isaac Lab backend value."""
    return value.torch if hasattr(value, "torch") else value


def _ground_support_measurement(contact_sensor: ContactSensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return force-weighted ground-contact centers, validity, forces, and contact positions."""
    ground_forces_w_n = _as_torch(contact_sensor.data.normal_force_matrix_w)[:, :, 0, :]
    ground_contact_positions_w_m = _as_torch(contact_sensor.data.contact_pos_w)[:, :, 0, :]
    normal_force_n = ground_forces_w_n[:, :, 2].clamp_min(0.0)
    total_normal_force_n = normal_force_n.sum(dim=1)
    finite_contact_positions = torch.isfinite(ground_contact_positions_w_m).all(dim=(1, 2))
    support_center_w_m = (ground_contact_positions_w_m.nan_to_num() * normal_force_n.unsqueeze(-1)).sum(dim=1)
    support_center_w_m /= total_normal_force_n.clamp_min(1.0).unsqueeze(-1)
    support_valid = finite_contact_positions & (total_normal_force_n >= args_cli.support_min_normal_force)
    return support_center_w_m, support_valid, ground_forces_w_n, ground_contact_positions_w_m


def run_smoke() -> dict[str, object]:
    """Simulate a commanded nominal pose and verify free-base standing observables."""
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.static_friction <= 0.0 or args_cli.dynamic_friction <= 0.0:
        raise ValueError("Contact friction coefficients must be positive.")
    if args_cli.log_file is not None:
        args_cli.log_file.with_suffix(".failure.txt").unlink(missing_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT_S, device=args_cli.device))
    contact_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=args_cli.static_friction,
        dynamic_friction=args_cli.dynamic_friction,
        restitution=0.0,
    )
    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=contact_material)
    ground_cfg.func("/World/Ground", ground_cfg)
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot_cfg = make_h1_2_cfg(args_cli.urdf, fixed_base=False)
    robot_cfg.spawn.physics_material = contact_material
    if args_cli.controller != "nominal":
        robot_cfg.actuators["legs"].damping = args_cli.leg_damping
        robot_cfg.actuators["feet"].stiffness = args_cli.foot_stiffness
        robot_cfg.actuators["feet"].damping = args_cli.foot_damping
    robot = Articulation(robot_cfg)
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/H1_2/.*_ankle_roll_link",
            update_period=0.0,
            history_length=4,
            filter_prim_paths_expr=[GROUND_COLLISION_PRIM_PATH],
            track_contact_points=True,
        )
    )
    sim.reset()

    joint_ids = resolve_joint_ids(robot.joint_names, ALL_JOINT_NAMES)
    targets = _as_torch(robot.data.default_joint_pos)[:, joint_ids].clone()
    for joint_name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        targets[:, ALL_JOINT_NAMES.index(joint_name)] += args_cli.shoulder_pitch_offset
    targets[:, ALL_JOINT_NAMES.index("left_shoulder_roll_joint")] += args_cli.left_shoulder_roll_offset
    targets[:, ALL_JOINT_NAMES.index("right_shoulder_roll_joint")] += args_cli.right_shoulder_roll_offset
    robot.write_joint_position_to_sim_index(position=targets, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(targets), joint_ids=joint_ids)
    robot.reset()
    initial_root_position_w_m = _as_torch(robot.data.root_pos_w)[0].clone()
    controller_cfg = StandingStabilizerCfg(
        pitch_kp=args_cli.ankle_pitch_kp,
        pitch_kd=args_cli.ankle_pitch_kd,
        hip_pitch_ratio=args_cli.hip_pitch_ratio,
        knee_ratio=args_cli.knee_ratio,
        root_position_kp=args_cli.root_position_kp,
        root_velocity_kd=args_cli.root_velocity_kd,
        support_position_kp=args_cli.support_position_kp,
        capture_point_kp=args_cli.capture_point_kp,
        shoulder_pitch_ratio=args_cli.shoulder_pitch_ratio,
        roll_kp=args_cli.ankle_roll_kp,
        roll_kd=args_cli.ankle_roll_kd,
        left_hip_roll_ratio=args_cli.left_hip_roll_ratio,
        right_hip_roll_ratio=args_cli.right_hip_roll_ratio,
        lateral_support_position_kp=args_cli.lateral_support_position_kp,
        lateral_capture_point_kp=args_cli.lateral_capture_point_kp,
        max_target_offset_rad=args_cli.ankle_target_limit,
        max_roll_target_offset_rad=args_cli.ankle_roll_target_limit,
    )
    trace_records: list[dict[str, object]] = []

    for step_index in range(args_cli.steps):
        step_targets = targets
        if args_cli.controller != "nominal":
            root_quaternion_w = _as_torch(robot.data.root_quat_w)
            pelvis_up_axis_w = quat_apply(
                root_quaternion_w,
                torch.tensor([[0.0, 0.0, 1.0]], device=sim.device),
            )
            step_targets = apply_standing_stabilizer(
                targets,
                ALL_JOINT_NAMES,
                pelvis_up_axis_w,
                _as_torch(robot.data.root_ang_vel_w),
                _as_torch(robot.data.root_pos_w),
                initial_root_position_w_m.unsqueeze(0),
                _as_torch(robot.data.root_lin_vel_w),
                controller_cfg,
                *_ground_support_measurement(contact_sensor)[:2],
            )
        robot.actuators.target_command.set_position_index(value=step_targets, joint_ids=joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT_S)
        contact_sensor.update(PHYSICS_DT_S)
        if args_cli.trace_file is not None:
            trace_root_quaternion_w = _as_torch(robot.data.root_quat_w)
            trace_up_axis_w = quat_apply(
                trace_root_quaternion_w,
                torch.tensor([[0.0, 0.0, 1.0]], device=sim.device),
            )[0]
            trace_contact_forces_w_n = _as_torch(contact_sensor.data.net_normal_forces_w)[0]
            support_center_w_m, support_valid, ground_forces_w_n, ground_contact_positions_w_m = _ground_support_measurement(
                contact_sensor
            )
            trace_records.append(
                {
                    "step": step_index + 1,
                    "time_s": (step_index + 1) * PHYSICS_DT_S,
                    "root_position_w_m": _as_torch(robot.data.root_pos_w)[0].tolist(),
                    "root_up_axis_w": trace_up_axis_w.tolist(),
                    "root_linear_velocity_w_mps": _as_torch(robot.data.root_lin_vel_w)[0].tolist(),
                    "root_angular_velocity_w_radps": _as_torch(robot.data.root_ang_vel_w)[0].tolist(),
                    "joint_position_targets_rad": step_targets[0].tolist(),
                    "foot_normal_force_norms_n": {
                        name: float(torch.linalg.vector_norm(trace_contact_forces_w_n[index]).item())
                        for index, name in enumerate(FOOT_BODY_NAMES)
                    },
                    "ground_contact_normal_forces_w_n": ground_forces_w_n[0].tolist(),
                    "ground_contact_positions_w_m": ground_contact_positions_w_m[0].tolist(),
                    "support_center_w_m": support_center_w_m[0].tolist(),
                    "support_center_valid": bool(support_valid[0].item()),
                }
            )

    root_position_w_m = _as_torch(robot.data.root_pos_w)[0]
    root_quaternion_w = _as_torch(robot.data.root_quat_w)[0]
    root_linear_velocity_w_mps = _as_torch(robot.data.root_lin_vel_w)[0]
    root_angular_velocity_w_radps = _as_torch(robot.data.root_ang_vel_w)[0]
    joint_position_rad = _as_torch(robot.data.joint_pos)[0, joint_ids]
    joint_velocity_radps = _as_torch(robot.data.joint_vel)[0, joint_ids]
    state_tensors = (
        root_position_w_m,
        root_quaternion_w,
        root_linear_velocity_w_mps,
        root_angular_velocity_w_radps,
        joint_position_rad,
        joint_velocity_radps,
    )
    if not all(torch.isfinite(value).all() for value in state_tensors):
        raise RuntimeError("Free-base H1-2 state became non-finite.")

    contact_forces_w_n = _as_torch(contact_sensor.data.net_normal_forces_w)[0]
    support_center_w_m, support_valid, ground_forces_w_n, ground_contact_positions_w_m = _ground_support_measurement(contact_sensor)
    contact_force_norms_n = {
        name: float(torch.linalg.vector_norm(contact_forces_w_n[index]).item())
        for index, name in enumerate(FOOT_BODY_NAMES)
    }
    up_axis_w = quat_apply(
        root_quaternion_w.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=sim.device)
    )[0]
    upright_cosine = float(up_axis_w[2].item())
    root_height_m = float(root_position_w_m[2].item())
    root_speed_mps = float(torch.linalg.vector_norm(root_linear_velocity_w_mps).item())
    root_angular_speed_radps = float(torch.linalg.vector_norm(root_angular_velocity_w_radps).item())
    result = {
        "status": "measured",
        "fixed_base": False,
        "merge_fixed_joints": False,
        "controller": args_cli.controller,
        "standing_stabilizer": {
            "pitch_kp": controller_cfg.pitch_kp,
            "pitch_kd": controller_cfg.pitch_kd,
            "hip_pitch_ratio": controller_cfg.hip_pitch_ratio,
            "knee_ratio": controller_cfg.knee_ratio,
            "root_position_kp": controller_cfg.root_position_kp,
            "root_velocity_kd": controller_cfg.root_velocity_kd,
            "support_position_kp": controller_cfg.support_position_kp,
            "capture_point_kp": controller_cfg.capture_point_kp,
            "shoulder_pitch_ratio": controller_cfg.shoulder_pitch_ratio,
            "roll_kp": controller_cfg.roll_kp,
            "roll_kd": controller_cfg.roll_kd,
            "left_hip_roll_ratio": controller_cfg.left_hip_roll_ratio,
            "right_hip_roll_ratio": controller_cfg.right_hip_roll_ratio,
            "lateral_support_position_kp": controller_cfg.lateral_support_position_kp,
            "lateral_capture_point_kp": controller_cfg.lateral_capture_point_kp,
            "max_target_offset_rad": controller_cfg.max_target_offset_rad,
            "max_roll_target_offset_rad": controller_cfg.max_roll_target_offset_rad,
        },
        "physics_dt_s": PHYSICS_DT_S,
        "implicit_pd": {
            "leg_damping_nm_s_per_rad": args_cli.leg_damping,
            "foot_stiffness_nm_per_rad": args_cli.foot_stiffness,
            "foot_damping_nm_s_per_rad": args_cli.foot_damping,
        },
        "shoulder_pitch_offset_rad": args_cli.shoulder_pitch_offset,
        "shoulder_roll_offsets_rad": {
            "left": args_cli.left_shoulder_roll_offset,
            "right": args_cli.right_shoulder_roll_offset,
        },
        "contact_friction": {
            "static": args_cli.static_friction,
            "dynamic": args_cli.dynamic_friction,
        },
        "steps": args_cli.steps,
        "simulated_duration_s": args_cli.steps * PHYSICS_DT_S,
        "root_position_w_m": root_position_w_m.tolist(),
        "root_displacement_w_m": (root_position_w_m - initial_root_position_w_m).tolist(),
        "root_height_m": root_height_m,
        "root_quaternion_w": root_quaternion_w.tolist(),
        "root_up_axis_w": up_axis_w.tolist(),
        "root_linear_velocity_w_mps": root_linear_velocity_w_mps.tolist(),
        "root_angular_velocity_w_radps": root_angular_velocity_w_radps.tolist(),
        "upright_cosine": upright_cosine,
        "root_linear_speed_mps": root_speed_mps,
        "root_angular_speed_radps": root_angular_speed_radps,
        "foot_normal_force_norms_n": contact_force_norms_n,
        "ground_contact_normal_forces_w_n": ground_forces_w_n[0].tolist(),
        "ground_contact_positions_w_m": ground_contact_positions_w_m[0].tolist(),
        "support_center_w_m": support_center_w_m[0].tolist(),
        "support_center_valid": bool(support_valid[0].item()),
        "finite_state": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args_cli.log_file is not None:
        args_cli.log_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.log_file.with_suffix(".measurement.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if args_cli.trace_file is not None:
        args_cli.trace_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.trace_file.write_text(json.dumps(trace_records, indent=2) + "\n", encoding="utf-8")
    if not 0.75 <= root_height_m <= 1.10:
        raise RuntimeError(f"Free-base root height {root_height_m} m is outside the standing range.")
    if upright_cosine < 0.90:
        raise RuntimeError(f"Free-base upright cosine {upright_cosine} is below 0.90.")
    if root_speed_mps > 0.50 or root_angular_speed_radps > 1.50:
        raise RuntimeError("Free-base H1-2 did not settle to the standing velocity bounds.")
    if any(force < 20.0 for force in contact_force_norms_n.values()):
        raise RuntimeError(f"Both feet must sustain contact, got {contact_force_norms_n}.")
    result["status"] = "passed"
    return result


def main() -> None:
    """Run the smoke test and optionally save its measurements."""
    result = run_smoke()
    output = json.dumps(result, indent=2)
    print(output)
    if args_cli.log_file is not None:
        args_cli.log_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.log_file.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if args_cli.log_file is not None:
            args_cli.log_file.parent.mkdir(parents=True, exist_ok=True)
            args_cli.log_file.with_suffix(".failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        simulation_app.close()
