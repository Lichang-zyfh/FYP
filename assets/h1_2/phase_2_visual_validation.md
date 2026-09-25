# Phase 2 grounded pose, contact, and joint-motion validation

## Result

Phase 2 fixed-base acceptance passed on 2026-09-21 after correcting the pelvis
height from `1.05 m` to `0.9625 m`. The old height left the shoe collision meshes
approximately `0.0875 m` above the ground and produced zero contact force, so the
earlier informal GUI judgment was superseded.

The corrected capture is in `phase_2_visual_evidence/` and contains the nominal
pose, a foot close-up, and positive-motion images for the right knee, torso,
right shoulder pitch, and right elbow. `evidence.json` records:

- left/right ankle-roll normal-force norms: `161.495 N` and `163.515 N`;
- positive right-knee motion: achieved `0.982173 rad`, local-axis rotation
  `0.117345 rad`, downstream displacement `0.111905 m`;
- positive torso motion: achieved `0.254746 rad`, local-axis rotation
  `0.253077 rad`;
- positive right-shoulder-pitch motion: achieved `0.460136 rad`, local-axis
  rotation `0.179577 rad`, downstream displacement `0.062687 m`;
- positive right-elbow motion: achieved `0.793074 rad`, local-axis rotation
  `0.211794 rad`, downstream displacement `0.024104 m`;
- finite articulation state throughout.

All requested targets remain inside their imported soft limits. The independent
validator additionally covers positive finite-difference direction for all 27
actuated joints, so the screenshots are representative evidence rather than the
only axis check.

## Asset acceptance decision

The ten links without individual collision geometry and the `logo_link` inertia
warning are accepted only for the fixed-base Phase 2/3 baseline. Six omitted
colliders are intermediate or terminal articulated links, three are empty sensor
frames, and `logo_link` is a fixed decorative visual. The official pinned URDF is
left unchanged to preserve its source hash.

This acceptance does not extend to free-base balance, self-collision safety, or
Sim-to-Real claims. Before free-base work, create a derived asset that either
adds justified collision/inertial data or merges the decorative fixed link, then
repeat collision, contact, and stability validation.

The measured foot forces confirm ground contact only. Because the pelvis is
fixed and its constraint shares external load, these forces must not be used as
dynamic whole-body ZMP evidence.
