# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""H1-2 whole-body standing and reaching task support."""

import gymnasium as gym

from h1_whole_body_reaching import agents

gym.register(
    id="H1-2-Whole-Body-Reaching-Direct-v0",
    entry_point="h1_whole_body_reaching.reaching_env:H1ReachingEnv",
    disable_env_checker=True,
)

gym.register(
    id="H1-2-Standing-PPO-Direct-v0",
    entry_point="h1_whole_body_reaching.reaching_env:H1ReachingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "h1_whole_body_reaching.reaching_env_cfg:make_h1_standing_env_cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1StandingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="H1-2-Standing-Push-PPO-Direct-v0",
    entry_point="h1_whole_body_reaching.reaching_env:H1ReachingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "h1_whole_body_reaching.reaching_env_cfg:make_h1_standing_push_env_cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1StandingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
