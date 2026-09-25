# H1-2 Phase 2/3 standing and reaching environment

This package is the validated implementation of the H1-2 Phase 2 asset checks
and Phase 3 pre-training reaching environment. It deliberately stops before
PPO, Advantage Mixing, and the B0 IK/whole-body-PD baseline. The current
environment is a vectorized `DirectRLEnv` with a 21-dimensional normalized
joint-position-offset action, a 66-dimensional policy observation, randomized
three-dimensional targets, shaped reaching reward, held-success termination,
timeout, and free-base fall/contact safety terminations.

## Current completion boundary

Phase 2 validated the checked-in H1-2 asset interface: all 27 joint names and
directions, joint limits, left/right symmetry, collision coverage, inertials,
independent URDF FK, foot contact, and static CoM support margin. Static CoM
projection is only a zero-acceleration ZMP equivalence check; it is not a
dynamic-ZMP claim.

Phase 3 provides repeatable reset, vectorized execution, target sampling within
the initial right-hand workspace, target visualization, action scaling,
observation/reward generation, and safety-aware episode accounting. The
free-base scenario suite runs static, arm-motion, short-push, injected-fall, and
injected-contact-loss cases and reports means, standard deviations, and 95% CIs.
It is a safety/diagnostic suite, not a push-recovery controller: recovering from
60 N or 120 N pushes remains future B0/control work.

Run static and interface validation first:

```bash
uv run python tools/validate_h1_2_description.py third_party/unitree_ros/robots/h1_2_description
uv run python -m pytest h1_whole_body_reaching/tests/test_interfaces.py
UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --with mujoco --with numpy \
  python tools/validate_h1_2_phase_2.py \
  third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  third_party/unitree_ros/robots/h1_2_description/h1_2_handless.xml
```

Then run one headless Isaac Lab environment for 1 s:

```bash
uv sync --extra isaacsim
uv run --no-sync python -m h1_whole_body_reaching.scripts.zero_action_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --steps 200 \
  --log-file logs/h1_2/phase_2_zero_action.json
```

The scene uses a fixed pelvis, explicit implicit-PD drive groups covering all 27 actuated joints, a 21-dimensional normalized position-offset policy interface, and nominal targets for the six excluded wrist joints.

The fixed pelvis is placed at `(0.0, 0.0, 0.9625) m`. This height was derived
from the nominal-pose URDF forward kinematics and the ankle-roll collision-mesh
minimum so that both soles contact the ground. Capture the maintained visual and
measured contact evidence with:

```bash
uv run --no-sync python -m h1_whole_body_reaching.scripts.phase_2_visual_evidence \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --output-dir assets/h1_2/phase_2_visual_evidence
```

The grounded capture measured `161.495 N` and `163.515 N` normal-force norms
at the left and right ankle-roll links and archived nominal, foot close-up, knee,
torso, shoulder, and elbow images. These forces establish contact only; they are
not dynamic-ZMP evidence because the fixed pelvis shares the external load.

The independent Phase 2 validator checks URDF inertials and collision coverage, all
27 joint finite-difference directions, left/right zero-pose symmetry, FK against
MuJoCo link origins, and a static CoM support margin. Its static ZMP equivalence is
valid only at zero acceleration and must not be used as a B4 dynamic-ZMP result.

Run the original fixed right-hand target probe after the standing rollout:

```bash
uv run --no-sync python -m h1_whole_body_reaching.scripts.fixed_target_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --log-file logs/h1_2/phase_3_fixed_target.json
```

The probe targets `right_wrist_yaw_link` at a fixed world-frame point, exposes a
six-value wrist-and-error observation, and succeeds only below a 0.01 m
position-error threshold. It remains useful for deterministic asset and marker
debugging, but is not the complete Phase 3 environment.

For bounded workspace diagnostics, the same proven rollout accepts `--target X Y Z`,
`--scripted-action` (21 values in frozen policy order), and
`--success-tolerance`. `--allow-miss` records rather than raises on a failed probe;
use it only to map the boundary, never as evidence that a target was reached. The
initial box was rejected after only five of eight corners met the 0.05 m
criterion. The revised fixed-base range is `x=[0.06, 0.10] m`,
`y=[-0.24, -0.19] m`, and `z=[0.8725, 0.9325] m`; all eight corners and its
center passed after the grounded-height correction, with final errors from
`0.011806 m` to `0.027835 m`.

Run the vectorized Phase 3 environment smoke test:

```bash
uv run --no-sync python -m h1_whole_body_reaching.scripts.smoke_reaching_env \
  --device cuda:0 \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf \
  --num-envs 16 --steps 100
```

Run the free-base scenario suite used for the current diagnostic evidence:

```bash
PYTHONUNBUFFERED=1 uv run python -m h1_whole_body_reaching.scripts.free_base_scenario_suite \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf \
  --device cuda:0 --steps 100 --seeds 7 11 17 \
  --push-forces 0 60 120 --push-duration-steps 10 --arm-ramp-steps 20
```

The suite requires static and arm-motion cases to survive their full horizon.
Push cases use a finite 0.2 s force and retain valid pre-terminal samples as
diagnostic evidence; a physical fall is reported, not silently discarded.

For deterministic fixed-base reset and termination contract checks, run:

```bash
uv run --no-sync python -m h1_whole_body_reaching.scripts.reset_termination_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --num-seeds 4 --seed 7 --settle-steps 100 \
  --log-file logs/h1_2/phase_4_reset_termination.json
```

Each fixed-base policy joint receives an independent deterministic offset in
`[-0.01, 0.01] rad`; the six wrist targets remain nominal. In the free-base
environment, reset uses the validated deterministic standing pose and also
checks low pelvis height, persistent tilt, and simultaneous loss of both foot
contacts.

If Kit stalls at `carb.omniclient.plugin`, verify that `readlink -f ~/.cache/ov` resolves to an existing writable directory. A dangling cache symlink can make OmniClient loop before renderer or physics initialization.
