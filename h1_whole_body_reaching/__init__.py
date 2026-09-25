# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""H1-2 whole-body standing and reaching task support."""

import gymnasium as gym

gym.register(
    id="H1-2-Whole-Body-Reaching-Direct-v0",
    entry_point="h1_whole_body_reaching.reaching_env:H1ReachingEnv",
    disable_env_checker=True,
)
