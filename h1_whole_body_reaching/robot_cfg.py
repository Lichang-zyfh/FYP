# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab configuration for the fixed-base Unitree H1-2 validation scene."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
ACTION_SCALE_RAD = 0.1
FIXED_BASE_ROOT_POSITION_W_M = (0.0, 0.0, 0.9625)


def make_h1_2_cfg(urdf_path: Path, *, fixed_base: bool = True, merge_fixed_joints: bool = False) -> ArticulationCfg:
    """Create an H1-2 articulation configuration.

    Args:
        urdf_path: H1-2 handless URDF path.
        fixed_base: Whether to constrain the pelvis to the world frame.
        merge_fixed_joints: Whether to merge fixed visual and sensor links at import.

    Returns:
        Articulation configuration with explicit position drives for all 27 joints.
    """
    if not urdf_path.is_file():
        raise FileNotFoundError(f"H1-2 URDF not found: {urdf_path}")
    hip_pitch_rad = -0.28 if fixed_base else -0.375
    knee_rad = 0.79 if fixed_base else 0.80
    ankle_pitch_rad = -0.52 if fixed_base else -0.425

    return ArticulationCfg(
        prim_path="/World/H1_2",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path.resolve()),
            fix_base=fixed_base,
            merge_fixed_joints=merge_fixed_joints,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=FIXED_BASE_ROOT_POSITION_W_M,
            joint_pos={
                ".*_hip_yaw_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_pitch_joint": hip_pitch_rad,
                ".*_knee_joint": knee_rad,
                ".*_ankle_pitch_joint": ankle_pitch_rad,
                ".*_ankle_roll_joint": 0.0,
                "torso_joint": 0.0,
                ".*_shoulder_pitch_joint": 0.28,
                ".*_shoulder_roll_joint": 0.0,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_joint": 0.52,
                ".*_wrist_.*_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
                joint_effort_limit={".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0},
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                },
                damping=5.0,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*_joint"],
                joint_effort_limit={".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0},
                stiffness=20.0,
                damping=4.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                joint_effort_limit=200.0,
                stiffness=200.0,
                damping=5.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
                joint_effort_limit={
                    ".*_shoulder_pitch_joint": 40.0,
                    ".*_shoulder_roll_joint": 40.0,
                    ".*_shoulder_yaw_joint": 18.0,
                    ".*_elbow_joint": 18.0,
                },
                stiffness=40.0,
                damping=10.0,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*_joint"],
                joint_effort_limit={
                    ".*_wrist_roll_joint": 19.0,
                    ".*_wrist_pitch_joint": 19.0,
                    ".*_wrist_yaw_joint": 19.0,
                },
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )
