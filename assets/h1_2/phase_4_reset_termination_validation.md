# Phase 4 reset randomization and termination validation

## Scope

This bounded phase adds deterministic reset perturbations and the first complete task-termination set to the fixed-base reaching prototype. It does not add rewards, PPO, target randomization, or a free-floating pelvis.

## Frozen reset contract

- Reset randomization applies only to the 21 policy joints, in the frozen `POLICY_JOINT_NAMES` order.
- A standard-library seeded generator samples each offset uniformly from `[-0.01, 0.01] rad`.
- The six excluded wrist targets remain at their nominal zero targets.
- This remains a direct fixed-base probe, rather than a manager-based environment, until the task has a reward and formal environment registration.

## Frozen termination contract

- `success`: right-hand target error is at most `0.01 m`.
- `time_out`: no earlier termination and completed control steps reach 200.
- `invalid_state`: target error or any joint position is non-finite.
- `joint_limit_violation`: any articulation position lies outside its imported hard joint limits.

Fall/contact termination is deliberately deferred: the pelvis is fixed in this prototype, so a torso-height fall condition would be meaningless. It must be added when the project releases the floating base.

## Evidence

The headless Isaac Sim smoke test passed on 2026-09-20 for four independent deterministic seeds, 7 through 10. Each reset settled for 100 physics steps (`0.5 s`) and had finite state, no hard-limit violation, and no timeout. The largest sampled policy offset was `0.00998257078325158 rad`; the largest settled position-target error was `0.04142874479293823 rad`. The output is stored in `logs/h1_2/phase_4_reset_termination.json`.

## Reproduction command

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  -u ROS_VERSION -u ROS_PYTHON_VERSION -u ROS_DOMAIN_ID \
  timeout 180 uv run --no-sync python -m h1_whole_body_reaching.scripts.reset_termination_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --num-seeds 4 --seed 7 --settle-steps 100 \
  --log-file logs/h1_2/phase_4_reset_termination.json
```
