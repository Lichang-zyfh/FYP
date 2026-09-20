# Phase 1 H1-2 asset validation

## Result

Phase 1 asset validation passed on 2026-09-20. The official Unitree H1-2 handless asset is reproducible from the pinned upstream commit recorded in `asset_manifest.json`; its static interface, direct MJCF dynamics, and Isaac Lab USD import were checked without changing `experiment_spec.md`.

## Confirmed interface

- Source: `unitree_ros` commit `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`, `robots/h1_2_description`, BSD-3-Clause.
- URDF: 32 links, 27 actuated joints, and `pelvis` as the sole root link.
- Frozen policy contract: 21 position-target actions in the order stored in `asset_manifest.json`; the six wrist joints are excluded.
- Candidate support links: `left_ankle_roll_link` and `right_ankle_roll_link`.
- Candidate right end-effector link: `right_wrist_yaw_link`.

## Runtime evidence

- Host verification: PyTorch CUDA is available on an NVIDIA GeForce RTX 4060 Laptop GPU with driver 595.84.
- Direct MuJoCo smoke test: the official handless MJCF loaded, ran for 1.0 s at 0.002 s timestep, retained finite state, exposed 27 actuators and 21 policy actuators, and reported eight contacts after stepping.
- Isaac Lab standalone importer: `isaacsim-asset-isolated==6.1.0.0` converted the official MJCF and fixed-base URDF into temporary USD assets. The fixed-base USD exposes 27 revolute joints, 21 non-wrist policy joints, a pelvis articulation root, and two fixed joints.

## Known limitations

- The converted MJCF USD warns that source motor declarations do not create usable physics-drive stiffness or damping. PD gains and drive behavior are therefore unverified and must be authored explicitly.
- The conversion warnings about checker textures, lights, sensors, and some material bindings are not control-interface failures, but visual and sensor fidelity remains out of scope until the task scene is built.
- No learning environment, target command, observation vector, PD action execution, or zero-action standing rollout exists yet. Phase 1 validates the asset boundary only.

## Failed attempts

The first URDF and MJCF conversion attempts used nonexistent `urdf/` and `mjcf/` subdirectories. Both converters initialize before checking the input path, so they consumed CPU without producing output before being stopped. The correct files are at the root of `robots/h1_2_description`; conversion then completed successfully.

## Next task

Create the Phase 2 one-environment fixed-base standing task. Explicitly configure all 27 drives, hold the six excluded wrist joints at their nominal targets, apply the frozen 21-action map, and verify reset plus a zero-action step before adding reach targets or PPO.
