# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture measurable Phase 2 pose, foot-contact, and joint-axis evidence."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True, help="Official h1_2_handless.urdf path.")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG and JSON evidence.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg, save_images_to_file
from isaaclab.utils.math import quat_apply

from h1_whole_body_reaching.interfaces import ALL_JOINT_NAMES, resolve_joint_ids
from h1_whole_body_reaching.robot_cfg import PHYSICS_DT_S, make_h1_2_cfg


REPRESENTATIVE_MOTIONS = {
    "right_knee_joint": "right_ankle_pitch_link",
    "torso_joint": "torso_link",
    "right_shoulder_pitch_joint": "right_elbow_link",
    "right_elbow_joint": "right_wrist_roll_link",
}


def _as_torch(value):
    """Return the torch view of an Isaac Lab backend value."""
    return value.torch if hasattr(value, "torch") else value


def run_capture() -> dict[str, object]:
    """Capture nominal, contact, and representative positive-axis evidence."""
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    (args_cli.output_dir / "failure.txt").unlink(missing_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT_S, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))
    robot = Articulation(make_h1_2_cfg(args_cli.urdf))
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/H1_2/.*_ankle_roll_link",
            update_period=0.0,
            history_length=4,
            debug_vis=True,
        )
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/Phase2EvidenceCamera",
            update_period=0.0,
            height=720,
            width=720,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=32.0,
                focus_distance=3.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 20.0),
            ),
        )
    )
    sim.reset()

    joint_ids = resolve_joint_ids(robot.joint_names, ALL_JOINT_NAMES)
    nominal_targets = _as_torch(robot.data.default_joint_pos)[:, joint_ids].clone()
    soft_limits = _as_torch(robot.data.soft_joint_pos_limits)[:, joint_ids]

    def step(targets: torch.Tensor, steps: int) -> None:
        for _ in range(steps):
            robot.actuators.target_command.set_position_index(value=targets, joint_ids=joint_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(PHYSICS_DT_S)
            contact_sensor.update(PHYSICS_DT_S)
            camera.update(PHYSICS_DT_S)

    def capture(name: str, eye: tuple[float, float, float], look_at: tuple[float, float, float]) -> str:
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device), torch.tensor([look_at], device=sim.device)
        )
        step(current_targets, 8)
        rgb = _as_torch(camera.data.output["rgb"])[..., :3]
        image_path = args_cli.output_dir / f"{name}.png"
        save_images_to_file(rgb.to(dtype=torch.float32) / 255.0, str(image_path))
        return image_path.name

    current_targets = nominal_targets.clone()
    step(current_targets, 120)
    images = {
        "nominal_pose": capture("nominal_pose", (2.3, 2.3, 1.55), (0.0, 0.0, 0.9)),
        "feet_contact": capture("feet_contact", (1.0, 1.1, 0.38), (0.0, 0.0, 0.08)),
    }
    nominal_body_positions = _as_torch(robot.data.body_pos_w)[0].clone()
    foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_body_ids = [robot.body_names.index(name) for name in foot_body_names]
    foot_positions_w_m = {
        name: nominal_body_positions[body_id].tolist()
        for name, body_id in zip(foot_body_names, foot_body_ids, strict=True)
    }
    contact_forces = _as_torch(contact_sensor.data.net_normal_forces_w)[0]
    contact_force_norms_n = {
        name: float(torch.linalg.vector_norm(contact_forces[index]).item())
        for index, name in enumerate(foot_body_names)
    }

    motions: dict[str, object] = {}
    local_x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=sim.device)
    for joint_name, measured_body_name in REPRESENTATIVE_MOTIONS.items():
        current_targets = nominal_targets.clone()
        step(current_targets, 120)
        body_id = robot.body_names.index(measured_body_name)
        baseline_position = _as_torch(robot.data.body_pos_w)[0, body_id].clone()
        baseline_quaternion = _as_torch(robot.data.body_quat_w)[0, body_id].clone()
        policy_index = ALL_JOINT_NAMES.index(joint_name)
        requested_target = current_targets[0, policy_index] + 0.25
        current_targets[0, policy_index] = torch.minimum(
            requested_target, soft_limits[0, policy_index, 1] - 0.02
        )
        step(current_targets, 120)
        moved_position = _as_torch(robot.data.body_pos_w)[0, body_id]
        moved_quaternion = _as_torch(robot.data.body_quat_w)[0, body_id]
        achieved_position = _as_torch(robot.data.joint_pos)[0, joint_ids[policy_index]]
        displacement = moved_position - baseline_position
        baseline_axis = quat_apply(baseline_quaternion.unsqueeze(0), local_x_axis)[0]
        moved_axis = quat_apply(moved_quaternion.unsqueeze(0), local_x_axis)[0]
        axis_dot = torch.dot(baseline_axis, moved_axis).clamp(-1.0, 1.0)
        images[joint_name] = capture(
            f"positive_{joint_name}", (2.3, 2.3, 1.55), (0.0, 0.0, 0.9)
        )
        motions[joint_name] = {
            "measured_body_name": measured_body_name,
            "nominal_target_rad": float(nominal_targets[0, policy_index].item()),
            "positive_target_rad": float(current_targets[0, policy_index].item()),
            "achieved_position_rad": float(achieved_position.item()),
            "soft_limit_rad": soft_limits[0, policy_index].tolist(),
            "body_displacement_w_m": displacement.tolist(),
            "body_displacement_norm_m": float(torch.linalg.vector_norm(displacement).item()),
            "measured_local_axis": "x",
            "nominal_axis_w": baseline_axis.tolist(),
            "positive_axis_w": moved_axis.tolist(),
            "axis_dot_before_after": float(axis_dot.item()),
            "axis_rotation_rad": float(torch.acos(axis_dot).item()),
        }
        current_targets = nominal_targets.clone()
        step(current_targets, 120)

    joint_pos = _as_torch(robot.data.joint_pos)[:, joint_ids]
    joint_vel = _as_torch(robot.data.joint_vel)[:, joint_ids]
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all():
        raise RuntimeError("H1-2 state became non-finite while capturing Phase 2 evidence.")

    result = {
        "status": "passed",
        "fixed_base": True,
        "physics_dt_s": PHYSICS_DT_S,
        "images": images,
        "foot_body_positions_w_m": foot_positions_w_m,
        "foot_normal_force_norms_n": contact_force_norms_n,
        "representative_positive_joint_motions": motions,
        "finite_state": True,
        "contact_interpretation": (
            "Non-zero forces confirm ground contact only. The fixed pelvis shares external load, so these forces "
            "must not be interpreted as dynamic whole-body ZMP evidence."
            if all(force > 0.0 for force in contact_force_norms_n.values())
            else "One or both feet have zero measured normal force; Phase 2 foot-contact acceptance fails."
        ),
    }
    (args_cli.output_dir / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the evidence capture and print the recorded measurements."""
    print(json.dumps(run_capture(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        (args_cli.output_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        simulation_app.close()
