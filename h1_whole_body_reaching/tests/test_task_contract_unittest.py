# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standard-library tests for Phase 3 task parameters and reward semantics."""

from __future__ import annotations

import unittest

from h1_whole_body_reaching.task_contract import ReachingTaskCfg, reaching_reward, validate_reaching_task_cfg


class ReachingTaskContractTest(unittest.TestCase):
    """Verify simulator-independent Phase 3 behavior."""

    def test_workspace_matches_validated_fixed_base_probe(self) -> None:
        """The sampling box remains inside the nine-point GPU-validated range."""
        cfg = ReachingTaskCfg()
        self.assertEqual(cfg.target_lower_w_m, (0.06, -0.24, 0.8725))
        self.assertEqual(cfg.target_upper_w_m, (0.10, -0.19, 0.9325))
        self.assertEqual(cfg.success_tolerance_m, 0.05)

    def test_reward_is_bounded_and_success_is_additive(self) -> None:
        """A closer end effector receives more reward, with an explicit success bonus."""
        cfg = ReachingTaskCfg()
        self.assertGreater(reaching_reward(0.01, cfg, False), reaching_reward(0.5, cfg, False))
        self.assertEqual(reaching_reward(0.0, cfg, True), cfg.reaching_reward_scale + cfg.success_reward)

    def test_workspace_rejects_reversed_bounds(self) -> None:
        """Target sampling requires a non-empty interval in every coordinate."""
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(target_lower_w_m=(0.1, -0.2, 1.0), target_upper_w_m=(0.1, -0.1, 1.1)))

    def test_free_base_safety_parameters_have_safe_defaults(self) -> None:
        """The free-base termination thresholds are explicit and physically ordered."""
        cfg = ReachingTaskCfg()
        self.assertEqual(cfg.free_base_fall_height_m, 0.75)
        self.assertEqual(cfg.free_base_min_upright_cosine, 0.90)
        self.assertEqual(cfg.free_base_tilt_grace_duration_s, 0.25)
        self.assertEqual(cfg.free_base_min_foot_contact_force_n, 20.0)
        self.assertEqual(cfg.free_base_contact_grace_duration_s, 0.10)

    def test_free_base_safety_rejects_invalid_thresholds(self) -> None:
        """Unsafe free-base termination thresholds are rejected before simulation."""
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(free_base_fall_height_m=0.0))
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(free_base_min_upright_cosine=1.1))
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(free_base_min_foot_contact_force_n=0.0))
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(free_base_contact_grace_duration_s=-0.01))
        with self.assertRaises(ValueError):
            validate_reaching_task_cfg(ReachingTaskCfg(free_base_tilt_grace_duration_s=-0.01))


if __name__ == "__main__":
    unittest.main()
