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
from h1_whole_body_reaching.task_contract import ReachingTaskCfg


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
