# Unitree H1-2 asset provenance

The project does not vendor the official description directory. Fetch it from Unitree's `unitree_ros` repository at commit `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee` and validate it before use:

```bash
git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros
git -C third_party/unitree_ros checkout ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee -- robots/h1_2_description
uv run python tools/validate_h1_2_description.py third_party/unitree_ros/robots/h1_2_description
uv run --with mujoco --with numpy python tools/smoke_h1_2_mujoco.py \
  third_party/unitree_ros/robots/h1_2_description/h1_2_handless.xml
```

The source is BSD-3-Clause licensed. Preserve its copyright and license text if any source asset is redistributed. `asset_manifest.json` records the exact upstream commit, checksums, frozen joint mapping, and Phase 2 controller parameters.

Phase 1 validated the static interface, direct MuJoCo dynamics, and standalone Isaac Lab conversion. Phase 2 adds the maintained fixed-base configuration and zero-action rollout under `h1_whole_body_reaching/`.

`derived/h1_2_handless_free_base.urdf` is a project-owned derivative for the next free-base gate. Generate it
from the pinned source with `uv run --no-sync python tools/derive_h1_2_free_base_asset.py <source> <output>`.
Its scope and validation status are recorded in `free_base_validation.md`.
