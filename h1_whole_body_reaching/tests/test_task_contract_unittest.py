# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standard-library tests for Phase 3 task parameters and reward semantics."""

from __future__ import annotations

import unittest

from h1_whole_body_reaching.task_contract import (
    ReachingTaskCfg,
    StandingTaskCfg,
    reaching_reward,
    target_box_corners_and_center,
    target_box_initial_errors_m,
    validate_reaching_task_cfg,
    validate_standing_task_cfg,
)


class ReachingTaskContractTest(unittest.TestCase):
    """Verify simulator-independent Phase 3 behavior."""

    def test_workspace_matches_validated_b0_probe(self) -> None:
        """The sampling box remains inside the B0-validated near workspace."""
        cfg = ReachingTaskCfg()
        self.assertEqual(cfg.target_lower_w_m, (0.1215, -0.221, 0.909))
        self.assertEqual(cfg.target_upper_w_m, (0.1225, -0.219, 0.911))
        self.assertEqual(cfg.success_tolerance_m, 0.05)

    def test_b0_target_box_has_no_initial_success_at_reference_wrist(self) -> None:
        """Every corner and center requires a nonzero B0 motion at reset."""
        cfg = ReachingTaskCfg()
        points = target_box_corners_and_center(cfg)
        errors_m = target_box_initial_errors_m((0.06877379, -0.21924412, 0.90927804), cfg)

        self.assertEqual(len(points), 9)
        self.assertTrue(all(error > cfg.success_tolerance_m for error in errors_m))

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

    def test_standing_reward_contract_excludes_reaching_parameters(self) -> None:
        """Phase 6 exposes only balance and action-smoothness reward terms."""
        cfg = StandingTaskCfg()
        self.assertEqual(cfg.fall_penalty, 5.0)
        self.assertGreater(cfg.upright_reward_weight, 0.0)
        self.assertGreater(cfg.base_height_reward_weight, 0.0)
        self.assertGreater(cfg.foot_contact_reward_weight, 0.0)
        self.assertGreater(cfg.posture_reward_weight, 0.0)
        self.assertGreater(cfg.action_smoothness_reward_weight, 0.0)
        self.assertFalse(hasattr(cfg, "reaching_reward_scale"))

    def test_standing_reward_rejects_invalid_scales_and_weights(self) -> None:
        """Invalid Phase 6 reward parameters fail before simulation launch."""
        with self.assertRaises(ValueError):
            validate_standing_task_cfg(StandingTaskCfg(fall_penalty=-1.0))
        with self.assertRaises(ValueError):
            validate_standing_task_cfg(StandingTaskCfg(posture_error_scale_rad=0.0))


if __name__ == "__main__":
    unittest.main()
