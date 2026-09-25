# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for repeated free-base evaluation statistics."""

import math

import pytest

from h1_whole_body_reaching.evaluation_statistics import summarize_scenario_records


def test_summarize_scenario_records_reports_sample_statistics() -> None:
    """Repeated scenario records expose rates and 95% confidence intervals."""
    records = [
        {"terminated": False, "truncated": False, "zmp_violation_ratio": 0.0, "minimum_support_margin_m": 0.1},
        {"terminated": False, "truncated": False, "zmp_violation_ratio": 0.1, "minimum_support_margin_m": 0.2},
        {"terminated": True, "truncated": False, "zmp_violation_ratio": 0.2, "minimum_support_margin_m": 0.3},
    ]

    summary = summarize_scenario_records(records)

    assert summary["sample_count"] == 3
    assert summary["survival_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["termination_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["zmp_violation_ratio"] == {
        "mean": pytest.approx(0.1),
        "std": pytest.approx(0.1),
        "ci95_half_width": pytest.approx(1.96 * 0.1 / math.sqrt(3.0)),
    }


def test_summarize_scenario_records_rejects_empty_input() -> None:
    """An empty evaluation cannot produce a statistical conclusion."""
    with pytest.raises(ValueError, match="At least one"):
        summarize_scenario_records([])
