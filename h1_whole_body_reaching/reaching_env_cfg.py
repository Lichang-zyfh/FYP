# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration factory for the Phase 3 reaching environment."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from h1_whole_body_reaching.robot_cfg import CONTROL_DECIMATION, PHYSICS_DT_S, make_h1_2_cfg
from h1_whole_body_reaching.task_contract import ReachingTaskCfg, StandingTaskCfg


@configclass
class H1ReachingSceneCfg(InteractiveSceneCfg):
    """H1-2 scene cloned once per vectorized environment."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot: ArticulationCfg = MISSING
    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/Ground/CollisionPlane"],
        track_contact_points=True,
    )
    prohibited_torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(pelvis|torso_link)",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/Ground/CollisionPlane"],
    )


@configclass
class H1ReachingEnvCfg(DirectRLEnvCfg):
    """Direct workflow config for Phase 3 reaching and free-base safety gates."""

    decimation = CONTROL_DECIMATION
    episode_length_s = 10.0
    action_space = 21
    observation_space = 66
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT_S, render_interval=CONTROL_DECIMATION)
    scene: H1ReachingSceneCfg = H1ReachingSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    task: ReachingTaskCfg = ReachingTaskCfg()
    standing_task: StandingTaskCfg = StandingTaskCfg()
    standing_only: bool = False
    standing_push_force_n: float = 0.0
    standing_push_start_step: int = 25
    standing_push_duration_steps: int = 10
    enable_standing_stabilizer: bool = True
    free_base_stabilizer: dict[str, float] = {
        "pitch_kp": 0.2,
        "pitch_kd": 0.05,
        "hip_pitch_ratio": 0.0,
        "knee_ratio": 0.0,
        "root_position_kp": 0.0,
        "root_velocity_kd": 0.0,
        "support_position_kp": 0.0,
        "capture_point_kp": 0.0,
        "roll_kp": 0.1,
        "roll_kd": 0.02,
        "left_hip_roll_ratio": 0.0,
        "right_hip_roll_ratio": 0.0,
        "lateral_support_position_kp": 0.0,
        "lateral_capture_point_kp": 0.0,
        "max_target_offset_rad": 0.35,
        "max_roll_target_offset_rad": 0.25,
    }
    free_base_shoulder_pitch_offset_rad: float = 0.45
    reset_joint_offset_bound_rad: float = 0.01
    fixed_base: bool = True


def make_h1_reaching_env_cfg(
    urdf_path: Path, num_envs: int, device: str, *, fixed_base: bool = True
) -> H1ReachingEnvCfg:
    """Create an explicitly asset-bound reaching configuration.

    Args:
        urdf_path: Official H1-2 handless URDF path.
        num_envs: Number of vectorized scenes.
        device: Isaac Lab simulation device identifier.
        fixed_base: Whether to constrain the pelvis to the world frame.

    Returns:
        Ready-to-instantiate direct environment configuration.
    """
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    cfg = H1ReachingEnvCfg()
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.fixed_base = fixed_base
    cfg.scene.robot = make_h1_2_cfg(urdf_path, fixed_base=fixed_base).replace(prim_path="{ENV_REGEX_NS}/Robot")
    if not fixed_base:
        cfg.reset_joint_offset_bound_rad = 0.0
        cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        cfg.scene.robot.actuators["feet"].stiffness = 200.0
        cfg.scene.robot.actuators["feet"].damping = 10.0
        cfg.scene.robot.actuators["legs"].damping = 10.0
        cfg.scene.robot.actuators["torso"].damping = 10.0
    return cfg


def make_b0_reaching_env_cfg(urdf_path: Path, num_envs: int, device: str) -> H1ReachingEnvCfg:
    """Create the free-base configuration used by the B0 IK/PD baseline.

    Args:
        urdf_path: Official H1-2 handless URDF path.
        num_envs: Number of vectorized scenes.
        device: Isaac Lab simulation device identifier.

    Returns:
        Free-base reaching configuration with B0 standing gains.
    """
    cfg = make_h1_reaching_env_cfg(urdf_path, num_envs, device, fixed_base=False)
    # B0 owns the right-arm pose. Do not add the Phase-4 diagnostic shoulder
    # offset, which changes its IK reference pose.
    cfg.free_base_shoulder_pitch_offset_rad = 0.0
    cfg.free_base_stabilizer["pitch_kp"] = 1.0
    cfg.free_base_stabilizer["pitch_kd"] = 0.15
    cfg.free_base_stabilizer["root_position_kp"] = -5.0
    cfg.free_base_stabilizer["root_velocity_kd"] = -1.0
    cfg.scene.robot.actuators["legs"].stiffness = 500.0
    cfg.scene.robot.actuators["legs"].damping = 25.0
    cfg.scene.robot.actuators["feet"].stiffness = 200.0
    cfg.scene.robot.actuators["feet"].damping = 20.0
    return cfg


def make_h1_standing_env_cfg(
    urdf_path: Path = Path("assets/h1_2/derived/h1_2_handless_free_base.urdf"),
    num_envs: int = 128,
    device: str = "cuda:0",
) -> H1ReachingEnvCfg:
    """Create the Phase 6 free-base PPO standing configuration.

    This factory keeps the frozen 21-action, 66-observation robot interface,
    but removes target sampling, reaching reward, and the deterministic
    standing stabilizer so PPO alone supplies the high-level action.

    Args:
        urdf_path: Derived free-base H1-2 URDF path.
        num_envs: Number of vectorized scenes.
        device: Isaac Lab simulation device identifier.

    Returns:
        Free-base configuration for the standing-only PPO prerequisite.
    """
    cfg = make_h1_reaching_env_cfg(urdf_path, num_envs, device, fixed_base=False)
    cfg.standing_only = True
    cfg.enable_standing_stabilizer = False
    cfg.free_base_shoulder_pitch_offset_rad = 0.0
    cfg.reset_joint_offset_bound_rad = 0.0
    return cfg


def make_h1_standing_push_env_cfg(
    urdf_path: Path = Path("assets/h1_2/derived/h1_2_handless_free_base.urdf"),
    num_envs: int = 128,
    device: str = "cuda:0",
) -> H1ReachingEnvCfg:
    """Create the final Phase 6 curriculum rung with a 20 N, 0.2 s pelvis push."""
    cfg = make_h1_standing_env_cfg(urdf_path, num_envs, device)
    cfg.standing_push_force_n = 20.0
    return cfg
