# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified-training registration hook for project-local Gym tasks."""


def register_h1_whole_body_tasks() -> list[str]:
    """Register H1-2 Gym tasks before Isaac Lab parses task metadata.

    Returns:
        No additional command-line arguments. The return value is consumed by
        Isaac Lab's ``--external_callback`` protocol.
    """
    import h1_whole_body_reaching  # noqa: F401

    return []
