# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the frozen Phase 4 metric contract."""

from __future__ import annotations

import json

import pytest

from h1_whole_body_reaching.phase_4_metrics import (
    Phase4EpisodeAccumulator,
    evaluate_joint_task_success,
    maximum_tolerated_disturbance_n,
    summarize_phase_4_records,
    write_phase_4_report,
)


@pytest.mark.parametrize(
    ("held", "fell", "support", "torso_contact", "expected"),
    [(True, False, True, False, True), (False, False, True, False, False), (True, True, True, False, False), (True, False, False, False, False), (True, False, True, True, False)],
)
def test_joint_success_requires_every_required_condition(held: bool, fell: bool, support: bool, torso_contact: bool, expected: bool) -> None:
    """Held reaching alone must never be reported as the formal task success."""
    assert evaluate_joint_task_success(held_reaching_success=held, fell=fell, bilateral_support_safe=support, prohibited_torso_contact=torso_contact) is expected


def _record(*, held_success: bool = True, fell: bool = False, zmp_outside: bool = False):
    accumulator = Phase4EpisodeAccumulator("nominal", 7, 0, 0.1)
    for action, velocity in (([0.0, 0.0], [0.0, 0.0]), ([0.1, 0.0], [1.0, 0.0]), ([0.2, 0.0], [3.0, 0.0])):
        accumulator.add_step(
            end_effector_error_m=0.04,
            torso_pitch_rad=0.1,
            torso_roll_rad=-0.2,
            com_support_margin_m=0.03,
            zmp_valid=True,
            zmp_outside_support=zmp_outside,
            zmp_deviation_m=0.01,
            action=action,
            joint_velocity_radps=velocity,
            applied_torque_nm=[2.0, 0.0],
            fell=fell,
            left_foot_safe=True,
            right_foot_safe=True,
            prohibited_torso_contact=False,
        )
    return accumulator.finalize(held_reaching_success=held_success, termination_cause="held_success")


def test_accumulator_computes_episode_metrics_and_safety_override() -> None:
    """Rates, effort, jerk, and a ZMP support failure are observable in the record."""
    record = _record(zmp_outside=True)
    assert not record.joint_task_success
    assert record.zmp_violation_ratio == pytest.approx(1.0)
    assert record.minimum_com_support_margin_m == pytest.approx(0.03)
    assert record.control_effort_j == pytest.approx(0.8)
    assert record.mean_action_rate_per_s == pytest.approx(1.0)
    assert record.peak_joint_jerk_radps3 == pytest.approx(100.0)


def test_terminal_fall_is_preserved_when_the_terminal_state_is_not_sampled() -> None:
    """A DirectRLEnv reset cannot erase the observable fall termination."""
    accumulator = Phase4EpisodeAccumulator("nominal", 7, 0, 0.1)
    accumulator.add_step(
        end_effector_error_m=0.04, torso_pitch_rad=0.0, torso_roll_rad=0.0, com_support_margin_m=0.03,
        zmp_valid=True, zmp_outside_support=False, zmp_deviation_m=0.01, action=[0.0],
        joint_velocity_radps=[0.0], applied_torque_nm=[0.0], fell=False, left_foot_safe=True,
        right_foot_safe=True, prohibited_torso_contact=False,
    )
    terminal_record = accumulator.finalize(held_reaching_success=False, termination_cause="fall")
    assert terminal_record.fall
    assert not terminal_record.joint_task_success


def test_summary_and_persistence_are_scenario_by_seed(tmp_path) -> None:
    """Synthetic episodes produce stable JSONL records and grouped confidence intervals."""
    success = _record()
    failure = _record(held_success=False, fell=True)
    summary = summarize_phase_4_records([success, failure])
    assert summary["joint_task_success_rate"] == pytest.approx(0.5)
    assert summary["fall_rate"] == pytest.approx(0.5)
    write_phase_4_report([success, failure], tmp_path, {"step_dt_s": 0.1})
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 2
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["scenarios"]["nominal"]["7"]["sample_count"] == 2


def test_maximum_tolerated_disturbance_requires_every_repeat_to_recover() -> None:
    """A single failed repeat prevents accepting that tested push magnitude."""
    recovered = _record()
    recovered = recovered.__class__(**{**recovered.__dict__, "disturbance_recovery_time_s": 0.5})
    failed_repeat = _record(held_success=False)
    assert maximum_tolerated_disturbance_n({20.0: [recovered], 40.0: [recovered, failed_repeat]}) == 20.0
    assert maximum_tolerated_disturbance_n({40.0: [failed_repeat]}) is None


def test_accumulator_rejects_misaligned_vectors() -> None:
    """A physics evaluator cannot silently combine different joint interfaces."""
    accumulator = Phase4EpisodeAccumulator("nominal", 7, 0, 0.1)
    with pytest.raises(ValueError, match="identical lengths"):
        accumulator.add_step(
            end_effector_error_m=0.1, torso_pitch_rad=0.0, torso_roll_rad=0.0, com_support_margin_m=None,
            zmp_valid=False, zmp_outside_support=False, zmp_deviation_m=None, action=[0.0],
            joint_velocity_radps=[0.0, 0.0], applied_torque_nm=[0.0], fell=False,
            left_foot_safe=True, right_foot_safe=True, prohibited_torso_contact=False,
        )
