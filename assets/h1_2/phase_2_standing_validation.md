# Phase 2 fixed-base standing validation

## Implemented contract

- Simulation timestep: 0.005 s; control decimation: 4; nominal control frequency: 50 Hz.
- Articulation: official `h1_2_handless.urdf`, fixed pelvis, ground plane, explicit implicit-PD drives for all 27 actuated joints.
- Policy interface: 21 normalized position-offset actions with scale 0.1 rad. The six wrist joints are excluded from the policy and held at their nominal 0 rad targets.
- Nominal pose: symmetric bent-leg pose with hips at -0.28 rad, knees at 0.79 rad, ankle pitch at -0.52 rad, shoulder pitch at 0.28 rad, and elbows at 0.52 rad.

## Evidence available

- Static URDF validation passed: 32 links, 27 actuated joints, pelvis root, exact 21-joint policy order, and valid joint limits.
- Direct MuJoCo smoke validation passed for 1.0 s at 0.002 s timestep with finite state and eight diagnostic contacts after stepping.
- Five unit tests for action expansion, wrist locking, and name ordering passed.
- Independent URDF-chain FK and MuJoCo checks passed for all 27 joints: the largest
  link-origin difference was `6.412880790956676e-08 m`; the nominal paired ankle and
  wrist geometry mirrored across the sagittal plane with `0.0 m` error. This check
  also reports a `66.984 kg` URDF total mass, collision geometry on both candidate
  support links, static CoM `(0.01055449, 0.00069932, 0.93987886) m`, and a lateral
  ankle-origin support margin of `0.16230068 m`. Ten auxiliary/intermediate links do
  not have their own collision geometry; that is an asset review finding, not proof
  that collision safety is complete.

## Isaac Sim rollout status

The 200-step zero-action Isaac Sim 6.1 rollout **passed** on the RTX 4060 Laptop GPU. The simulation completed 1.0 s with finite state, all 27 actuated joints, the frozen 21-joint policy interface, six 0 rad wrist targets, and a maximum nominal-pose error of 0.046060919761657715 rad. The machine-readable output is `logs/h1_2/phase_2_zero_action.json`.

## GUI smoke validation

The visible Isaac Lab GUI smoke validation **passed** after enabling a persistent 16 GiB swap file and launching through NVIDIA PRIME render offload. The scene rendered on the RTX 4060 Laptop GPU, completed 20 steps (0.1 s), emitted `status: passed`, reported finite state, and shut down normally after 92.698 s. Its maximum nominal-pose error was 0.02624887228012085 rad. The machine-readable output is `logs/h1_2/phase_2_gui_smoke.json`.

This verifies the graphical launch path and fixed-base rollout. A detailed manual record of pose appearance, foot contact, and joint-axis orientation is still needed before claiming those individual visual subchecks.

## Resolved startup failure

The startup failure was not caused by the 8 GiB GPU. `/home/zlc/.cache/ov` was a dangling symbolic link to an unavailable external-drive path, `/media/zlc/T7/isaac_env/ov_cache`. `carb.omniclient.plugin` repeatedly received `EEXIST` from `mkdir` and `ENOENT` from `stat`, consuming one CPU core indefinitely. The link was preserved as `/home/zlc/.cache/ov.broken-link-20260920` and replaced with a local cache directory; an empty Isaac Sim app then started in about 12 s.

The later GUI hang had a separate resource cause: kernel logs showed `nvidia_uvm` failed an order-9 allocation while the 16 GiB system had no swap. A persistent 16 GiB `/swapfile` was enabled, and the GUI smoke command used `__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and `__VK_LAYER_NV_optimus=NVIDIA_only` under the X11 on-demand graphics profile.

## Reproduction command

```bash
uv sync --extra isaacsim
uv run --no-sync python -m h1_whole_body_reaching.scripts.zero_action_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --steps 200 \
  --log-file logs/h1_2/phase_2_zero_action.json
```

The headless command emits a `status: passed` JSON result. The recorded GUI smoke command also passed; record a detailed visual pose/contact/axis review before treating every Phase 2 visual subcheck as closed.

## Independent Phase 2 asset checks

```bash
UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --with mujoco --with numpy \
  python tools/validate_h1_2_phase_2.py \
  third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  third_party/unitree_ros/robots/h1_2_description/h1_2_handless.xml
```

The static ZMP statement in this check is deliberately limited to static equilibrium:
with zero acceleration, the ZMP equals the CoM projection. It is not a dynamic ZMP
implementation or validation.
