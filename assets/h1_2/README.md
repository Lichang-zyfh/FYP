# Unitree H1-2 asset provenance

The project does not vendor the 80 MiB official description directory. Fetch it from Unitree's `unitree_ros` repository at commit `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee` and validate it before use:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros
git -C third_party/unitree_ros sparse-checkout set robots/h1_2_description
python tools/validate_h1_2_description.py third_party/unitree_ros/robots/h1_2_description
```

The source is BSD-3-Clause licensed. Preserve its copyright and license text if the URDF, MJCF, or meshes are copied into this repository. `asset_manifest.json` records the exact upstream commit and checksums for the handless URDF and MJCF used by the FYP.

Phase 1 validation confirmed the source model's 27 actuated joints and the 21-joint policy order formed by excluding both three-DoF wrists. On an RTX 4060 host, the handless MJCF loaded and stepped for 1.0 s in MuJoCo without a non-finite state; it also retained eight contacts after stepping. Isaac Lab's standalone importer generated both an MJCF USD and a fixed-base URDF USD, each with 27 revolute joints and a pelvis articulation root.

The importer reports that the MJCF motor declarations do not create usable USD drive stiffness or damping. Do not use the converted USD for control until Phase 2 defines the PD drives, nominal pose, contact configuration, and fixed wrist behavior explicitly. See `phase_1_asset_validation.md` and `asset_manifest.json` for the frozen evidence and known limitations.
