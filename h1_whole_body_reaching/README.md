# H1-2 Whole-Body Standing and Reaching Environment

This package implements the H1-2 asset validation, reaching environment,
Phase 4 evaluation, Phase 5 B0 IK/PD baseline, and Phase 6 standing PPO
prerequisite. It is a vectorized `DirectRLEnv` with a frozen 21-dimensional
normalized joint-position-offset action and 66-dimensional policy observation.

## Phase 4 to 6 completion summary

- **Phase 4:** Versioned JSONL evaluation records and scenario-by-seed summaries
  measure reaching success, falls, bilateral support, torso contact, CoM/ZMP
  diagnostics, action rate, joint jerk, control effort, and finite-push recovery.
  The PPO evaluator loads RSL-RL checkpoints while keeping environment stepping
  outside `torch.inference_mode()`, which permits reliable multi-seed resets.
- **Phase 5:** The B0 free-base baseline combines damped-least-squares right-arm
  IK with position-PD joint targets. It is a deterministic diagnostic baseline,
  not an RL policy, and uses the same Phase 4 report schema.
- **Phase 6:** `H1-2-Standing-PPO-Direct-v0` removes target sampling and reaching
  reward, disables the deterministic standing stabilizer, and rewards only
  uprightness, root height, bilateral foot contact, nominal posture, action
  smoothness, and fall/contact-loss avoidance. The push curriculum task applies
  a finite 0.2 s world-x pelvis force after 25 control steps.

The completed Phase 6 curriculum progressed through 5 N, 10 N, and 20 N
pushes. Final checkpoint:

```text
logs/rsl_rl/h1_2_standing_ppo/2026-09-26_13-40-56_phase_6_push_20n_refine_seed7/model_995.pt
```

For seeds 7, 11, and 17, this checkpoint completed 100 control steps without a
fall in both static standing and a 20 N, 0.2 s world-x pelvis-push evaluation.
Every push trial reached the evaluator's 0.5 s post-push stable window. This is
standing-only evidence; it must not be reported as reaching success.

Run the final Phase 6 acceptance checks with:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --extra rsl_rl python \
  -m h1_whole_body_reaching.scripts.evaluate_phase_4 \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf --device cuda:0 \
  --controller ppo_standing --checkpoint <model_995.pt> \
  --steps 100 --seeds 7 11 17 --output-dir /tmp/phase6-static

PYTHONPATH=. UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --extra rsl_rl python \
  -m h1_whole_body_reaching.scripts.evaluate_phase_4 \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf --device cuda:0 \
  --controller ppo_standing --checkpoint <model_995.pt> \
  --steps 100 --seeds 7 11 17 --push-force-n 20 --push-duration-steps 10 \
  --output-dir /tmp/phase6-push-20n
```

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

## Phase 6 standing PPO implementation

`H1-2-Standing-PPO-Direct-v0` is deliberately separate from the reaching task.
It keeps the frozen 21 normalized joint-position-offset actions and 66 policy
values, replacing the three target-relative values with pelvis-up-axis values.
Its reward contains only uprightness, root-height maintenance, bilateral foot
contact, nominal posture, action smoothness, and a fall/contact-loss penalty.
It neither samples a target nor rewards reaching, and it disables the B0-style
standing stabilizer so PPO supplies every high-level action.

The external callback makes the project-local Gym registration visible to Isaac
Lab's unified training command. Use `H1-2-Standing-Push-PPO-Direct-v0` only for
the 5 N → 10 N → 20 N curriculum; keep the static task for its regression test.

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --extra rsl_rl isaaclab train \
  --rl_library rsl_rl \
  --external_callback h1_whole_body_reaching.registration.register_h1_whole_body_tasks \
  --task H1-2-Standing-PPO-Direct-v0 --device cuda:0 --num_envs 128 \
  --seed 7 --max_iterations 10 --run_name phase_6_smoke

PYTHONPATH=. UV_CACHE_DIR=/tmp/isaaclab-uv-cache uv run --extra rsl_rl isaaclab train \
  --rl_library rsl_rl \
  --external_callback h1_whole_body_reaching.registration.register_h1_whole_body_tasks \
  --task H1-2-Standing-PPO-Direct-v0 --device cuda:0 --num_envs 128 \
  --seed 7 --checkpoint logs/rsl_rl/h1_2_standing_ppo/<run>/model_10.pt \
  --max_iterations 2000 --run_name phase_6_resume
```

Use the existing Phase 4 evaluator only after adding a trained-policy action
source; the current `zero` and `b0` action sources are not evidence about PPO.
Evaluate the final selected checkpoints under seeds 7, 11, and 17 for static,
scripted arm-motion, and 0.2 s world-x push scenarios before calling the Phase
6 acceptance gate complete.

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

## Phase 4 evaluation

Phase 4 records one versioned JSONL object per episode and a scenario-by-seed
JSON summary. Formal joint success requires the 0.5 s held reaching condition,
no fall, continuous bilateral safe support (including a valid in-support ZMP),
and no pelvis or torso ground contact. It separately reports final and mean
end-effector error, survival time, torso pitch/roll, CoM support margin, ZMP
violation/deviation, action rate, joint jerk, and integrated absolute joint
power. A finite push additionally records recovery time only after 0.5 s of
post-push stable support; missing recovery remains `null`, not a fabricated
zero.

Run a fresh, deterministic evaluation (not a training rollout) with:

```bash
PYTHONUNBUFFERED=1 uv run python -m h1_whole_body_reaching.scripts.evaluate_phase_4 \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf --device cuda:0 \
  --steps 500 --seeds 7 11 17 --episodes-per-seed 1 \
  --output-dir logs/h1_2/phase_4_nominal
```

Use `--push-force-n 60 --push-duration-steps 10` for the repeatable 0.2 s
disturbance protocol. The evaluator measures a policy; it does not imply that
the zero-action standing controller has achieved reaching or push recovery.
Evaluate the B0 IK/PD baseline with the same JSONL schema by selecting its
controller and a fixed world-frame target. A 0.3 s ramp is the validated B0
setting for the narrow target box:

```bash
PYTHONUNBUFFERED=1 uv run python -m h1_whole_body_reaching.scripts.evaluate_phase_4 \
  --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf --device cuda:0 \
  --controller b0 --target 0.122 -0.220 0.910 --target-ramp-s 0.3 \
  --scenario-name b0_fixed_center --steps 100 --seeds 7 11 17 \
  --output-dir logs/h1_2/phase_4_b0_fixed_center
```

With a nonzero push, B0 continues through the requested horizon after its
held-reaching event so the evaluator can measure the required post-push 0.5 s
recovery interval. Its `joint_task_success` still requires that event, no
fall, bilateral safe support, and no prohibited torso contact.
Use `--case held_success`, `timeout`, `fall`, `lost_contact`, and
`zmp_outside_support` to exercise the corresponding deterministic acceptance
paths before training. The last case is an explicitly synthetic metric-path
injection: it preserves the measured foot-support geometry and supplies a ZMP
beyond its edge, so it validates the rejection rule without claiming a
physical disturbance produced that ZMP. To report the maximum tolerated push,
call `maximum_tolerated_disturbance_n` with records grouped by tested force;
a magnitude is accepted only when every repeat both meets joint success and
completes the 0.5 s recovery hold.

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
