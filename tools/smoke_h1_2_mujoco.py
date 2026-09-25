#!/usr/bin/env python3
"""Run a minimal dynamic smoke test on the official H1-2 handless MJCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


EXCLUDED_WRISTS = frozenset(
    {
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
)


def run_smoke_test(mjcf_path: Path, duration_s: float) -> dict[str, object]:
    """Load and step the MJCF model without control.

    Args:
        mjcf_path: Official H1-2 handless MJCF file.
        duration_s: Simulated duration [s].

    Returns:
        Machine-readable validation result.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
    data = mujoco.MjData(model)
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    ]
    policy_actuators = [name for name in actuator_names if name not in EXCLUDED_WRISTS]
    if model.nu != 27:
        raise RuntimeError(f"Expected 27 actuators, got {model.nu}.")
    if len(policy_actuators) != 21:
        raise RuntimeError(f"Expected 21 policy actuators, got {len(policy_actuators)}.")

    step_count = round(duration_s / model.opt.timestep)
    for _ in range(step_count):
        mujoco.mj_step(model, data)
    if not np.isfinite(np.concatenate((data.qpos, data.qvel))).all():
        raise RuntimeError("Model state became non-finite during the smoke test.")
    return {
        "status": "passed",
        "timestep_s": model.opt.timestep,
        "simulated_duration_s": data.time,
        "actuator_count": model.nu,
        "policy_actuator_count": len(policy_actuators),
        "contact_count_after_step": data.ncon,
        "root_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0),
        "finite_state": True,
    }


def main() -> None:
    """Run the command-line smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjcf", type=Path, help="Path to h1_2_handless.xml")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulated duration [s].")
    args = parser.parse_args()
    print(json.dumps(run_smoke_test(args.mjcf, args.duration), indent=2))


if __name__ == "__main__":
    main()
