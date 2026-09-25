# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the frozen H1-2 action interface."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as xml_etree

import pytest

from h1_whole_body_reaching.interfaces import (
    ALL_JOINT_NAMES,
    POLICY_JOINT_NAMES,
    WRIST_JOINT_NAMES,
    expand_policy_action,
    resolve_joint_ids,
)
from h1_whole_body_reaching.task_contract import (
    RIGHT_END_EFFECTOR_BODY_NAME,
    FixedRightHandTargetCfg,
    ResetRandomizationCfg,
    apply_reset_policy_joint_offsets,
    evaluate_target_termination,
    evaluate_task_termination,
    make_reset_policy_joint_offsets,
    make_scripted_target_action,
    make_target_observation,
    target_error_norm_m,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_interface_matches_manifest() -> None:
    """The code contract must match the versioned asset manifest."""
    manifest = json.loads((REPOSITORY_ROOT / "assets/h1_2/asset_manifest.json").read_text(encoding="utf-8"))
    static_interface = manifest["static_interface"]
    assert list(POLICY_JOINT_NAMES) == static_interface["policy_joint_order"]
    assert list(WRIST_JOINT_NAMES) == static_interface["excluded_wrist_joints"]
    assert len(ALL_JOINT_NAMES) == static_interface["actuated_joint_count"]


def test_derived_free_base_asset_repairs_declared_collisions() -> None:
    """The free-base asset retains portable meshes and fixes the declared collision gaps."""
    asset_path = REPOSITORY_ROOT / "assets/h1_2/derived/h1_2_handless_free_base.urdf"
    root = xml_etree.parse(asset_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    repaired_links = (
        "left_hip_yaw_link",
        "left_ankle_pitch_link",
        "right_hip_yaw_link",
        "right_ankle_pitch_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    for link_name in repaired_links:
        assert len(links[link_name].findall("collision")) == 1
    logo_mass = links["logo_link"].find("inertial/mass")
    assert logo_mass is not None
    assert float(logo_mass.attrib["value"]) == pytest.approx(0.001)
    assert all(
        not mesh.attrib["filename"].startswith("/")
        for mesh in root.findall(".//mesh")
    )


def test_zero_action_keeps_all_nominal_targets() -> None:
    """Zero normalized action maps to the nominal 27-joint pose."""
    nominal = [0.01 * index for index in range(len(ALL_JOINT_NAMES))]
    targets = expand_policy_action([0.0] * len(POLICY_JOINT_NAMES), nominal, action_scale=0.1)
    assert targets == nominal


def test_policy_action_never_changes_wrist_targets() -> None:
    """The six excluded wrist joints remain fixed at nominal targets."""
    nominal = [0.0] * len(ALL_JOINT_NAMES)
    targets = expand_policy_action([1.0] * len(POLICY_JOINT_NAMES), nominal, action_scale=0.1)
    for wrist_name in WRIST_JOINT_NAMES:
        assert targets[ALL_JOINT_NAMES.index(wrist_name)] == 0.0
    for policy_name in POLICY_JOINT_NAMES:
        assert targets[ALL_JOINT_NAMES.index(policy_name)] == pytest.approx(0.1)


def test_action_range_and_shape_are_enforced() -> None:
    """Invalid policy vectors fail before reaching the simulator."""
    nominal = [0.0] * len(ALL_JOINT_NAMES)
    with pytest.raises(ValueError, match="Expected 21"):
        expand_policy_action([0.0] * 20, nominal, action_scale=0.1)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        expand_policy_action([2.0] + [0.0] * 20, nominal, action_scale=0.1)


def test_joint_ids_follow_requested_order() -> None:
    """Runtime indices are resolved without assuming importer order."""
    reversed_names = list(reversed(ALL_JOINT_NAMES))
    ids = resolve_joint_ids(reversed_names, POLICY_JOINT_NAMES)
    assert [reversed_names[index] for index in ids] == list(POLICY_JOINT_NAMES)


def test_fixed_target_observation_and_termination_contract() -> None:
    """The fixed target has a six-value observation and unambiguous termination."""
    target_cfg = FixedRightHandTargetCfg()
    observation = make_target_observation(target_cfg.position_w_m, target_cfg)
    assert RIGHT_END_EFFECTOR_BODY_NAME == "right_wrist_yaw_link"
    assert observation == (*target_cfg.position_w_m, 0.0, 0.0, 0.0)
    assert target_error_norm_m(observation) == 0.0
    assert evaluate_target_termination(0.0, target_cfg.episode_length_steps, target_cfg).success
    assert not evaluate_target_termination(0.0, target_cfg.episode_length_steps, target_cfg).time_out
    assert evaluate_target_termination(0.02, target_cfg.episode_length_steps, target_cfg).time_out


def test_scripted_target_action_preserves_non_right_shoulder_joints() -> None:
    """The deterministic probe changes only the intended policy coordinate."""
    action = make_scripted_target_action()
    right_shoulder_pitch_index = POLICY_JOINT_NAMES.index("right_shoulder_pitch_joint")
    assert len(action) == len(POLICY_JOINT_NAMES)
    assert action[right_shoulder_pitch_index] == pytest.approx(-0.5)
    assert sum(value != 0.0 for value in action) == 1


def test_reset_offsets_are_deterministic_bounded_and_preserve_wrists() -> None:
    """Reset randomization is reproducible and does not alter excluded wrists."""
    reset_cfg = ResetRandomizationCfg(policy_joint_position_offset_bound_rad=0.01)
    first = make_reset_policy_joint_offsets(seed=7, reset_cfg=reset_cfg)
    second = make_reset_policy_joint_offsets(seed=7, reset_cfg=reset_cfg)
    targets = apply_reset_policy_joint_offsets([0.0] * len(ALL_JOINT_NAMES), first)

    assert first == second
    assert all(abs(offset) <= reset_cfg.policy_joint_position_offset_bound_rad for offset in first)
    for wrist_name in WRIST_JOINT_NAMES:
        assert targets[ALL_JOINT_NAMES.index(wrist_name)] == 0.0


def test_task_termination_prioritizes_invalid_and_limit_states() -> None:
    """Safety states override success and timeout in the initial task contract."""
    target_cfg = FixedRightHandTargetCfg()
    nominal = [0.0] * len(ALL_JOINT_NAMES)
    lower = [-1.0] * len(ALL_JOINT_NAMES)
    upper = [1.0] * len(ALL_JOINT_NAMES)

    success = evaluate_task_termination(0.0, 1, target_cfg, nominal, lower, upper)
    timeout = evaluate_task_termination(0.02, target_cfg.episode_length_steps, target_cfg, nominal, lower, upper)
    invalid = evaluate_task_termination(float("nan"), 1, target_cfg, nominal, lower, upper)
    exceeded = evaluate_task_termination(0.0, 1, target_cfg, [1.1, *nominal[1:]], lower, upper)

    assert success.success and not success.time_out
    assert timeout.time_out and not timeout.success
    assert invalid.invalid_state and not invalid.success and not invalid.time_out
    assert exceeded.joint_limit_violation and not exceeded.success and not exceeded.time_out
