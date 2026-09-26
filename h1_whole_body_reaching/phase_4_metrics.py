# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned per-episode metrics for Phase 4 evaluation.

The functions in this module deliberately depend only on the Python standard
library.  This makes the task-success rule and report calculations testable
without Isaac Sim, while the evaluator supplies measurements from the running
environment.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev


PHASE_4_METRICS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Phase4EpisodeRecord:
    """Persisted metric record for one evaluation episode.

    All angular quantities are in [rad], distances are in [m], durations are
    in [s], forces are in [N], and control effort is in [J].
    """

    schema_version: str
    scenario: str
    seed: int
    episode_index: int
    termination_cause: str
    held_reaching_success: bool
    joint_task_success: bool
    fall: bool
    bilateral_support_safe: bool
    prohibited_torso_contact: bool
    final_end_effector_error_m: float
    mean_end_effector_error_m: float
    survival_time_s: float
    mean_torso_pitch_rad: float
    mean_torso_roll_rad: float
    peak_abs_torso_pitch_rad: float
    peak_abs_torso_roll_rad: float
    minimum_com_support_margin_m: float
    zmp_violation_ratio: float
    maximum_zmp_deviation_m: float
    mean_action_rate_per_s: float
    peak_joint_jerk_radps3: float
    control_effort_j: float
    disturbance_recovery_time_s: float | None


def evaluate_joint_task_success(
    *,
    held_reaching_success: bool,
    fell: bool,
    bilateral_support_safe: bool,
    prohibited_torso_contact: bool,
) -> bool:
    """Return the frozen Phase 4 joint reaching-and-balance success predicate."""
    return held_reaching_success and not fell and bilateral_support_safe and not prohibited_torso_contact


class Phase4EpisodeAccumulator:
    """Accumulate one episode's metrics from time-aligned evaluator samples."""

    def __init__(self, scenario: str, seed: int, episode_index: int, step_dt_s: float) -> None:
        if not scenario:
            raise ValueError("scenario must not be empty.")
        if step_dt_s <= 0.0 or not math.isfinite(step_dt_s):
            raise ValueError("step_dt_s must be finite and positive.")
        self.scenario = scenario
        self.seed = seed
        self.episode_index = episode_index
        self.step_dt_s = step_dt_s
        self._errors_m: list[float] = []
        self._pitch_rad: list[float] = []
        self._roll_rad: list[float] = []
        self._com_margins_m: list[float] = []
        self._zmp_deviations_m: list[float] = []
        self._zmp_valid_samples = 0
        self._zmp_violations = 0
        self._action_rates_per_s: list[float] = []
        self._joint_jerks_radps3: list[float] = []
        self._control_effort_j = 0.0
        self._previous_action: tuple[float, ...] | None = None
        self._previous_joint_velocity: tuple[float, ...] | None = None
        self._previous_joint_acceleration: tuple[float, ...] | None = None
        self._fell = False
        self._bilateral_support_safe = True
        self._prohibited_torso_contact = False
        self._post_push_stable_steps: int | None = None
        self._recovery_time_s: float | None = None

    def add_step(
        self,
        *,
        end_effector_error_m: float,
        torso_pitch_rad: float,
        torso_roll_rad: float,
        com_support_margin_m: float | None,
        zmp_valid: bool,
        zmp_outside_support: bool,
        zmp_deviation_m: float | None,
        action: Sequence[float],
        joint_velocity_radps: Sequence[float],
        applied_torque_nm: Sequence[float],
        fell: bool,
        left_foot_safe: bool,
        right_foot_safe: bool,
        prohibited_torso_contact: bool,
        disturbance_active: bool = False,
        recovery_hold_duration_s: float = 0.5,
    ) -> None:
        """Add one post-step sample.

        ``zmp_deviation_m`` is the horizontal ZMP distance from the current
        force-weighted support center.  Jerk is the peak norm of the finite
        difference of joint acceleration, and control effort is the integrated
        absolute mechanical joint power.
        """
        scalars = (end_effector_error_m, torso_pitch_rad, torso_roll_rad, recovery_hold_duration_s)
        if any(not math.isfinite(value) for value in scalars) or end_effector_error_m < 0.0:
            raise ValueError("Scalar metric inputs must be finite and error non-negative.")
        if len(action) != len(joint_velocity_radps) or len(action) != len(applied_torque_nm):
            raise ValueError("Action, velocity, and torque vectors must have identical lengths.")
        vectors = (action, joint_velocity_radps, applied_torque_nm)
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Vector metric inputs must be finite.")
        self._errors_m.append(end_effector_error_m)
        self._pitch_rad.append(torso_pitch_rad)
        self._roll_rad.append(torso_roll_rad)
        if com_support_margin_m is not None:
            if not math.isfinite(com_support_margin_m):
                raise ValueError("com_support_margin_m must be finite when supplied.")
            self._com_margins_m.append(com_support_margin_m)
        if zmp_valid:
            self._zmp_valid_samples += 1
            self._zmp_violations += int(zmp_outside_support)
            if zmp_deviation_m is not None:
                if not math.isfinite(zmp_deviation_m) or zmp_deviation_m < 0.0:
                    raise ValueError("zmp_deviation_m must be finite and non-negative when supplied.")
                self._zmp_deviations_m.append(zmp_deviation_m)
        action_tuple = tuple(action)
        velocity_tuple = tuple(joint_velocity_radps)
        if self._previous_action is not None:
            squared_difference = sum((current - previous) ** 2 for current, previous in zip(action_tuple, self._previous_action, strict=True))
            self._action_rates_per_s.append(math.sqrt(squared_difference) / self.step_dt_s)
        if self._previous_joint_velocity is not None:
            acceleration = tuple(
                (current - previous) / self.step_dt_s
                for current, previous in zip(velocity_tuple, self._previous_joint_velocity, strict=True)
            )
            if self._previous_joint_acceleration is not None:
                squared_jerk = sum(
                    ((current - previous) / self.step_dt_s) ** 2
                    for current, previous in zip(acceleration, self._previous_joint_acceleration, strict=True)
                )
                self._joint_jerks_radps3.append(math.sqrt(squared_jerk))
            self._previous_joint_acceleration = acceleration
        self._control_effort_j += sum(abs(torque * velocity) for torque, velocity in zip(applied_torque_nm, velocity_tuple, strict=True)) * self.step_dt_s
        self._previous_action = action_tuple
        self._previous_joint_velocity = velocity_tuple
        self._fell |= fell
        self._bilateral_support_safe &= left_foot_safe and right_foot_safe and (not zmp_outside_support if zmp_valid else False)
        self._prohibited_torso_contact |= prohibited_torso_contact
        stable = not fell and left_foot_safe and right_foot_safe and not prohibited_torso_contact and zmp_valid and not zmp_outside_support
        if disturbance_active:
            self._post_push_stable_steps = 0
        elif self._post_push_stable_steps is not None and self._recovery_time_s is None:
            self._post_push_stable_steps = self._post_push_stable_steps + 1 if stable else 0
            if self._post_push_stable_steps >= math.ceil(recovery_hold_duration_s / self.step_dt_s):
                self._recovery_time_s = self._post_push_stable_steps * self.step_dt_s

    def finalize(self, *, held_reaching_success: bool, termination_cause: str) -> Phase4EpisodeRecord:
        """Build the immutable per-episode record after an episode ends."""
        if not self._errors_m:
            raise ValueError("Cannot finalize an episode without metric samples.")
        if not termination_cause:
            raise ValueError("termination_cause must not be empty.")
        # DirectRLEnv resets a terminated scene before returning from step().
        # Evaluators therefore retain the last physical sample rather than
        # recording the reset pose; a terminal fall must still be persisted.
        fell = self._fell or termination_cause == "fall"
        return Phase4EpisodeRecord(
            schema_version=PHASE_4_METRICS_SCHEMA_VERSION,
            scenario=self.scenario,
            seed=self.seed,
            episode_index=self.episode_index,
            termination_cause=termination_cause,
            held_reaching_success=held_reaching_success,
            joint_task_success=evaluate_joint_task_success(
                held_reaching_success=held_reaching_success,
                fell=fell,
                bilateral_support_safe=self._bilateral_support_safe,
                prohibited_torso_contact=self._prohibited_torso_contact,
            ),
            fall=fell,
            bilateral_support_safe=self._bilateral_support_safe,
            prohibited_torso_contact=self._prohibited_torso_contact,
            final_end_effector_error_m=self._errors_m[-1],
            mean_end_effector_error_m=mean(self._errors_m),
            survival_time_s=len(self._errors_m) * self.step_dt_s,
            mean_torso_pitch_rad=mean(self._pitch_rad),
            mean_torso_roll_rad=mean(self._roll_rad),
            peak_abs_torso_pitch_rad=max(abs(value) for value in self._pitch_rad),
            peak_abs_torso_roll_rad=max(abs(value) for value in self._roll_rad),
            minimum_com_support_margin_m=min(self._com_margins_m, default=0.0),
            zmp_violation_ratio=self._zmp_violations / self._zmp_valid_samples if self._zmp_valid_samples else 0.0,
            maximum_zmp_deviation_m=max(self._zmp_deviations_m, default=0.0),
            mean_action_rate_per_s=mean(self._action_rates_per_s) if self._action_rates_per_s else 0.0,
            peak_joint_jerk_radps3=max(self._joint_jerks_radps3, default=0.0),
            control_effort_j=self._control_effort_j,
            # A transient stable window is not recovery if the episode later
            # terminates in a fall.
            disturbance_recovery_time_s=None if fell else self._recovery_time_s,
        )


def summarize_phase_4_records(records: Sequence[Phase4EpisodeRecord]) -> dict[str, object]:
    """Return scenario aggregate rates and sample statistics with 95% CIs."""
    if not records:
        raise ValueError("At least one Phase 4 episode record is required.")

    def continuous(attribute: str) -> dict[str, float]:
        samples = [float(getattr(record, attribute)) for record in records]
        deviation = stdev(samples) if len(samples) > 1 else 0.0
        return {"mean": mean(samples), "std": deviation, "ci95_half_width": 1.96 * deviation / math.sqrt(len(samples))}

    return {
        "sample_count": len(records),
        "joint_task_success_rate": mean(float(record.joint_task_success) for record in records),
        "fall_rate": mean(float(record.fall) for record in records),
        "post_push_success_rate": mean(float(record.joint_task_success) for record in records if record.disturbance_recovery_time_s is not None)
        if any(record.disturbance_recovery_time_s is not None for record in records)
        else None,
        "final_end_effector_error_m": continuous("final_end_effector_error_m"),
        "mean_end_effector_error_m": continuous("mean_end_effector_error_m"),
        "survival_time_s": continuous("survival_time_s"),
        "minimum_com_support_margin_m": continuous("minimum_com_support_margin_m"),
        "zmp_violation_ratio": continuous("zmp_violation_ratio"),
        "maximum_zmp_deviation_m": continuous("maximum_zmp_deviation_m"),
        "mean_action_rate_per_s": continuous("mean_action_rate_per_s"),
        "peak_joint_jerk_radps3": continuous("peak_joint_jerk_radps3"),
        "control_effort_j": continuous("control_effort_j"),
    }


def maximum_tolerated_disturbance_n(records_by_force_n: Mapping[float, Sequence[Phase4EpisodeRecord]]) -> float | None:
    """Return the greatest tested push magnitude with post-push success.

    A magnitude is tolerated only if every repeat at that magnitude reached
    the frozen joint-success condition and completed the recovery hold.  The
    function intentionally returns a tested value, not an interpolation.

    Args:
        records_by_force_n: Evaluation records keyed by push magnitude [N].

    Returns:
        The greatest fully tolerated tested magnitude [N], or ``None`` when
        no tested magnitude was tolerated.
    """
    if not records_by_force_n:
        raise ValueError("At least one tested push magnitude is required.")
    tolerated: list[float] = []
    for force_n, records in records_by_force_n.items():
        if not math.isfinite(force_n) or force_n < 0.0:
            raise ValueError("Push magnitudes must be finite and non-negative.")
        if not records:
            raise ValueError("Every tested push magnitude must have at least one record.")
        if all(record.joint_task_success and record.disturbance_recovery_time_s is not None for record in records):
            tolerated.append(force_n)
    return max(tolerated, default=None)


def write_phase_4_report(records: Sequence[Phase4EpisodeRecord], output_dir: Path, config: Mapping[str, object]) -> None:
    """Persist JSONL episode records and scenario-by-seed JSON summaries."""
    if not records:
        raise ValueError("At least one Phase 4 episode record is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    grouped: dict[str, dict[str, list[Phase4EpisodeRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.scenario][str(record.seed)].append(record)
    report = {
        "schema_version": PHASE_4_METRICS_SCHEMA_VERSION,
        "config": dict(config),
        "scenarios": {
            scenario: {seed: summarize_phase_4_records(seed_records) for seed, seed_records in seeds.items()}
            for scenario, seeds in grouped.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
