# H1-2 derived free-base asset validation

## Derived asset

`derived/h1_2_handless_free_base.urdf` is generated from the pinned Unitree H1-2 handless URDF by
`tools/derive_h1_2_free_base_asset.py`. The official source remains unchanged.

- Six movable links that had visual meshes but no collision geometry now use a matching mesh collision:
  both hip-yaw links, both ankle-pitch links, and both wrist-yaw links.
- `logo_link` now has a fixed-link inertial of `0.001 kg` and a positive-definite diagonal inertia of
  `1e-8 kg m^2`; no collision is added for the decorative logo.
- Mesh references are relative to the derived asset and resolve to the pinned source mesh directory.
- The derived asset keeps all 32 source links and the 27 actuated joints. It is imported for free-base
  checks with `fix_base=False`, `merge_fixed_joints=False`, and self-collision still disabled.

## Static validation

The derived-asset test in `h1_whole_body_reaching/tests/test_interfaces.py` passed on 2026-09-21.
It parsed the URDF, verified each repaired collision, verified the logo mass, and rejected absolute mesh paths.

## Runtime status

The intended runtime gate is:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  -u ROS_VERSION -u ROS_PYTHON_VERSION -u ROS_DOMAIN_ID \
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  __VK_LAYER_NV_optimus=NVIDIA_only \
  uv run --no-sync python -m h1_whole_body_reaching.scripts.free_base_standing_smoke \
  --visualizer none --device cuda:0 --urdf assets/h1_2/derived/h1_2_handless_free_base.urdf --steps 500 \
  --log-file logs/h1_2/free_base_standing_smoke.json
```

The host-GPU reset/contact gate passed on 2026-09-21: after 10 steps (0.05 s), the root height was 0.96020818 m,
the upright cosine was 0.99999958, and the left/right ankle-roll normal-force norms were 472.111 N and 464.777 N.
The required 500-step (2.5 s) standing gate failed under a fixed nominal-pose PD command: root height was
0.10997 m, upright cosine was 0.02045, and both ankle contacts were lost. The asset import, 27 joint commands,
and contact sensor are therefore validated; a feedback standing controller is required before this becomes a
free-base balance-training environment. A bounded symmetric ankle-pitch feedback experiment is implemented in
`h1_whole_body_reaching.standing_controller` and selected with `--controller ankle`. Its reverse-direction
candidate reduced 100-step forward drift to 0.05071 m and angular speed to 0.63269 rad/s, but still fell by
500 steps; it is diagnostic evidence rather than a passing standing controller. A coupled hip/knee/ankle
candidate (`--controller sagittal`, pitch gains `-1.0` and `-0.15`, ratios `-0.5` and `0.25`) stayed upright
for 100 steps but fell by 500. Adding captured-initial-pose forward root position/velocity feedback confirmed
the corrective sign: gains `-0.4 rad/m` and `-0.1 s/m` reduced the 100-step forward drift from 0.05064 m to
0.04679 m, but the 500-step result was still a fall (root height 0.08315 m, upright cosine 0.05862, and no
foot contact). These bounded feedback controllers remain diagnostics, not a passing standing controller. The
prior silent run was caused by launching without `--visualizer none`, not by host permissions.

The controller now also supports bounded bilateral ankle-roll feedback (`--controller whole_body`) and
optional per-step JSON traces (`--trace-file`). Because both ankle-roll axes point along `+X`, the same
roll correction is applied to both joints. The confirmed corrective gains (`-1.0` and `-0.15`) reduced
100-step lateral displacement from 0.002819 m without roll feedback to 0.002737 m, and reduced the
500-step displacement from 0.021457 m to 0.018743 m. The 500-step gate still failed, so lateral drift is
secondary rather than the standing failure's main cause.

The 500-step trace shows growing forward velocity before the fall: the default 0.20 rad ankle-pitch
correction saturated at 1.05 s, upright cosine dropped below 0.90 at 1.19 s, root height dropped below
0.75 m at 1.29 s, and both foot contacts were lost after 1.505 s. Raising the correction limit to 0.35 rad
(a target of -0.87 rad, within the -0.897 rad joint limit) delayed the upright-threshold crossing by only
0.025 s and did not delay the height failure. This rules out the original correction limit as the primary
cause: the next controller iteration needs earlier support-aware sagittal feedback instead of another
gain/limit-only sweep.
