# Phase 1 H1-2 asset validation

## Result

Phase 1 passed on 2026-09-20. The official Unitree H1-2 handless asset is reproducible from the pinned upstream commit in `asset_manifest.json`. The frozen experiment specification was not changed.

## Confirmed interface

- Official source: `unitree_ros` commit `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`, directory `robots/h1_2_description`, BSD-3-Clause.
- URDF: 32 links, 27 actuated joints, and `pelvis` as the sole root link.
- Policy contract: 21 normalized position-offset actions; both three-DoF wrists are excluded and held at nominal targets.
- Candidate support links: `left_ankle_roll_link` and `right_ankle_roll_link`.
- Candidate right end-effector link: `right_wrist_yaw_link`.

## Runtime evidence from the first session

- The handless MJCF stepped for 1.0 s at 0.002 s timestep with finite state, 27 actuators, 21 policy actuators, and eight contacts after stepping. Contact count is diagnostic evidence, not a frozen interface contract.
- Isaac Lab standalone importers converted the MJCF and fixed-base URDF. The fixed-base USD exposed 27 revolute joints, 21 policy joints, a pelvis articulation root, and two fixed joints.

## Recovered first-session gap

The first session wrote its executable validation scripts under `/tmp` and committed the project records from a temporary minimal repository. Its Git branch therefore contains the project records but not the full Isaac Lab source tree. Those scripts are historical evidence only; maintained Phase 1 and Phase 2 code now lives in this repository.

## Failed attempts

The first URDF and MJCF conversion attempts used nonexistent `urdf/` and `mjcf/` subdirectories. The files are at the root of `robots/h1_2_description`; conversion succeeded after correcting the path.
