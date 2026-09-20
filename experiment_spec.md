# H1-2 Whole-Body Reaching Experiment Specification

**Version:** 1.0
**Status:** frozen on 2026-09-20
**Authority:** This specification governs the Phase 0--4 implementation and evaluation. A change to a frozen item needs supervisor approval or recorded experimental evidence, an incremented version, and an entry in the experiment log. An implementation parameter that is explicitly marked **asset-validation gate** is not a silent exception: it must be filled in once, committed with the asset-validation evidence, and then becomes frozen.

## 1. Research question and scope

The study asks whether one unified PPO policy can coordinate single-arm three-dimensional reaching and whole-body standing balance on Unitree H1-2, and whether Advantage Mixing, a CoM-based balance reward, or a ZMP-based balance reward improves that coordination. The target arm is the **right arm** in the first study. The policy may move the non-reaching joints to preserve balance; the task does not ask the left arm to reach.

Each episode starts from a bilateral standing posture on a level plane. The robot receives a position-only right end-effector target and must reach it while preserving safe contact and posture. Walking, stepping, grasping, target perception, wrist/dexterous-hand control, uneven terrain, and hardware deployment are out of scope for the core comparison. They may only be added as separately versioned extensions after B0--B4 on this task are complete.

The primary outcome is joint-task success, not reaching error alone. An episode succeeds only when all of the following hold continuously for **0.5 s** before its 10 s timeout:

- right end-effector position error is at most **0.05 m**;
- neither foot has left its prescribed support region;
- no fall or prohibited torso/ground contact occurred; and
- torso roll and pitch remain within the safety limits recorded with the H1-2 asset manifest.

The 0.05 m and 0.5 s thresholds are the frozen reporting contract. Asset-derived joint and contact limits are safety constraints, not tunable reward weights.

## 2. Robot, interfaces, and time base

The experimental robot is **Unitree H1-2**. The supplied H1-2 dynamic-balance paper reports 27 DoF and a 21-DoF policy that excludes the three wrist DoF of each hand. Accordingly, this study freezes a **21-dimensional normalized joint-position-offset action** for all learned methods. The six wrist joints are held at the nominal pose by the low-level controller. The policy-controlled set contains the two legs, waist, and both non-wrist arms; the right arm is the only reaching arm and all remaining controlled joints constitute the balance group.

At each 50 Hz control step, action `a in [-1, 1]^21` is converted to `q_target = q_nominal + scale * a` and tracked by the asset's PD actuator model. Physics runs at **200 Hz** (`dt = 0.005 s`, decimation 4). The target is expressed in a base-fixed reference frame whose origin and axes are saved in the asset manifest; observations use the same frame. A target is sampled only from the collision-free, IK-reachable set defined by that manifest. This avoids treating an arbitrary Cartesian box as reachable.

**Asset-validation gate.** The checked-in Isaac Lab configuration references `Robots/Unitree/H1/h1.usd` from Nucleus, not a verified H1-2 model. Before a formal rollout, record the exact USD/URDF checksum and source, 27 joint names and order, 21 controlled indices, wrist indices, limits, nominal pose, link masses/inertias, foot and torso body names, end-effector frame, actuator gains/limits, and correspondence to the real robot interface. If this gate fails, the experiment is blocked rather than silently substituted with H1.

The actor observation is proprioceptive only: base orientation (projected gravity), base angular velocity, all controlled joint positions and velocities, bilateral foot contact/wrench state, right end-effector-to-target vector, previous action, and a phase-free target-valid flag. The critic may additionally consume the same simulation-only dynamics quantities needed to calculate CoM/ZMP, but no vision. Observation order, scales, and action-to-joint mapping must be emitted to a versioned manifest and shared by B1--B4.

## 3. Fixed methods and learning contract

All learned groups use the same policy inputs, action interface, PPO implementation, actor/critic capacity, simulator settings, randomization schedule, number of environments, total environment-step budget, checkpoint cadence, and evaluation suite. PPO uses clipped probability ratios; any implementation must log the clip fraction, approximate KL, entropy, value loss, and branch advantages. The final total training budget is an asset/platform-validation gate, but the same committed value must be used for every learned group.

| ID | Frozen method | Only intended difference |
| --- | --- | --- |
| B0 | Standing controller plus right-arm IK and PD | Non-RL engineering baseline |
| B1 | Vanilla unified PPO | Total advantage for all 21 actions |
| B2 | B1 plus linear Advantage Mixing | Policy-gradient advantage assignment |
| B3 | B2 plus CoM balance reward | CoM balance term replaces B2's base balance term |
| B4 | B2 plus ZMP balance reward | ZMP balance term replaces B2's base balance term |

For B1--B4, rewards are partitioned into `r_reach` (end-effector progress and terminal reaching success) and `r_balance` (survival, contact/posture safety, and the selected balance objective); identical regularizers for action rate, energy/control effort, joint limits, and self-collision remain outside the ablation. Returns and GAE advantages are computed per branch. B1 applies standard PPO to `A_total = A_reach + A_balance`. B2--B4 use the supplied Deep Whole-Body Control rule with `beta(t) = min(t / T_mix, 1)`, where `T_mix = 0.2 * N_train` environment steps:

`A_right_arm = A_reach + beta * A_balance`
`A_balance_group = A_balance + beta * A_reach`

The right-arm log-probability uses `A_right_arm`; the other 17 action dimensions use `A_balance_group`. Each branch is normalized independently before mixing. PPO ratios and clipping remain per action dimension/group and must not be replaced by a second optimizer. The critic implementation must expose separately testable reach and balance value estimates; B1 retains the same critic capacity but does not mix its actor advantage.

B3 measures balance with a CoM projection/support-margin objective. B4 measures ZMP from contact forces and moments (or a documented dynamics-equivalent formulation) and evaluates its support-margin violation. CoM projection must never be labeled or substituted as ZMP. Static standing, prescribed right-arm motion, horizontal push, and single-foot-lift tests are mandatory unit-level validations of the chosen CoM/ZMP implementation before B3/B4 training.

## 4. Training, evaluation, and reporting

Difficulty is staged but method-identical: S0 fixed reachable frontal target with nominal dynamics; S1 randomized reachable targets; S2 adds end-effector payload variation; S3 adds external pushes and dynamics randomization. Target workspace, payload, push, friction, mass/CoM, actuator-gain/delay, and observation-noise ranges are each **asset-validation gates**. They are set once in configuration files after the asset manifest passes and are then used unchanged for every learned method. No training episode is a reported test result.

The formal evaluation suite contains nominal target, randomized in-distribution target, payload, push, and held-out dynamics cases. Every method is trained with **five seeds** and evaluated with **100 fresh episodes per seed per scenario** using fixed, published evaluation seeds. Report mean, standard deviation, and 95% confidence interval across training seeds for: joint-task success rate; final and mean end-effector error; fall rate and survival time; CoM support margin; ZMP violation ratio and maximum deviation (where applicable); disturbance recovery time; peak action rate/joint jerk; control effort; and environment steps to the predeclared success threshold. Comparisons use the same checkpoint-selection rule, selected only from a validation scenario, with final results produced on the held-out suite.

Each run must save the Git commit, spec version, asset checksum, configuration, random seeds, hardware/GPU metadata, training logs, selected checkpoint, and evaluator output. A failure is a result: retain the termination reason and scenario parameters instead of discarding failed episodes or reporting only the best seed.

## 5. Phase gates and non-negotiable limits

Phase 0 is complete when this file and the project-context handoff are committed. Phase 1 must validate the exact H1-2 asset and emit the interface manifest; it does not change the core task, B0--B4 definitions, success contract, or evaluation protocol. It may fill only the marked asset-validation fields. Code must not begin B1--B4 training, claim a Sim-to-Real result, or substitute the repository H1 locomotion task for this standing-reaching task before that gate passes.

References: Fu, Cheng, and Pathak, *Deep Whole-Body Control* (Advantage Mixing); Schulman et al., *PPO* (clipped objective); Ferigo et al., *Whole-Body Strategies* (push-recovery evaluation); Xie et al., *Dynamic Balance and RL* (H1-2 27/21-DoF and ZMP validation boundary).
