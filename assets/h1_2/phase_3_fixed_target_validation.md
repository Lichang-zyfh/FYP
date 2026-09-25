# Phase 3 fixed right-hand target validation

## Scope

This phase introduces a validated fixed target and a narrow fixed-base target-sampling box for the registered Direct environment. It does not add PPO, Advantage Mixing, dynamic balance rewards, or a free base.

## Frozen contract

- End effector: `right_wrist_yaw_link`.
- Fixed target position: `(0.0843, -0.2163, 0.9063) m` in the fixed-base world frame.
- Observation: six values: right-wrist world position `[m]` and target-relative error `[m]`.
- Action: existing 21-value normalized policy vector with scale `0.1 rad`; the deterministic probe sets only `right_shoulder_pitch_joint` to `-0.5`, equivalent to a `-0.05 rad` target offset.
- Termination: success at Euclidean error `<= 0.01 m`; otherwise timeout after 200 scripted-control steps.

## Evidence

The grounded-height headless Isaac Sim rollout passed on 2026-09-21. After 200 nominal-pose settling steps, the right-hand error was `0.01993394444085502 m`, confirming that the target was not already satisfied. After 200 scripted-action steps, the error was `0.003453159602370314 m`; the success flag was true and the timeout flag was false. The result is stored in `logs/h1_2/phase_3_fixed_target.json`.

## Reproduction command

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  -u ROS_VERSION -u ROS_PYTHON_VERSION -u ROS_DOMAIN_ID \
  uv run --no-sync python -m h1_whole_body_reaching.scripts.fixed_target_rollout \
  --device cuda:0 \
  --urdf third_party/unitree_ros/robots/h1_2_description/h1_2_handless.urdf \
  --log-file logs/h1_2/phase_3_fixed_target.json
```

For visual inspection, add `--visualizer kit` and the PRIME render-offload variables recorded in `asset_manifest.json`. The red sphere marker denotes the fixed target. The systematic Phase 2 capture in `phase_2_visual_evidence/` supersedes the earlier informal GUI judgment and confirms the corrected grounded pose with measured bilateral contact forces.

## Validated fixed-base workspace

The first provisional box, `x=[0.04, 0.12] m`, `y=[-0.26, -0.17] m`, and
`z=[0.94, 1.04] m`, was rejected because only five of its eight corners met
the `0.05 m` criterion. The replacement box is:

- `x=[0.06, 0.10] m`;
- `y=[-0.24, -0.19] m`;
- `z=[0.8725, 0.9325] m`.

On 2026-09-21, all eight corners and the center passed separate host-GPU
rollouts using bounded 21-dimensional actions after correcting the fixed-base
height. Initial errors ranged from `0.015249 m` to `0.053830 m`, and final
errors ranged from `0.011806 m` to `0.027835 m`. This validates only the
fixed-base sampling box and does not
establish a free-base whole-body reachable workspace.

The per-point targets, four non-zero arm-action fields, and initial/final errors
are archived in `phase_3_workspace_evidence.json`.

The registered environment was then run with two environments and zero actions
for 30 control steps. Its 66-value observations and rewards remained finite;
the reward range was `[0.958969, 2.971657]`, two held-success terminations were
observed, and no timeout occurred.
