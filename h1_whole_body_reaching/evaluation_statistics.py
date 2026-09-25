# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Statistics helpers for repeatable free-base evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import mean, stdev


def summarize_scenario_records(records: list[Mapping[str, float | bool]]) -> dict[str, object]:
    """Summarize repeated free-base evaluation records.

    Args:
        records: Per-seed records containing termination state and the scalar
            ``zmp_violation_ratio`` and ``minimum_support_margin_m`` metrics.

    Returns:
        Aggregate sample statistics and a normal-approximation 95% confidence
        interval half width for each continuous metric.

    Raises:
        ValueError: If no records are provided.
    """
    if not records:
        raise ValueError("At least one evaluation record is required.")

    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records]

    def summary(key: str) -> dict[str, float]:
        samples = values(key)
        deviation = stdev(samples) if len(samples) > 1 else 0.0
        return {
            "mean": mean(samples),
            "std": deviation,
            "ci95_half_width": 1.96 * deviation / math.sqrt(len(samples)),
        }

    return {
        "sample_count": len(records),
        "survival_rate": mean(float(not record["terminated"] and not record["truncated"]) for record in records),
        "termination_rate": mean(float(record["terminated"]) for record in records),
        "zmp_violation_ratio": summary("zmp_violation_ratio"),
        "minimum_support_margin_m": summary("minimum_support_margin_m"),
    }
