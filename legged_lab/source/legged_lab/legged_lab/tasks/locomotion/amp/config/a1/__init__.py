import gymnasium as gym

from legged_lab.envs import ManagerBasedAmpEnv

from . import agents

gym.register(
    id="LeggedLab-Isaac-AMP-A1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Fixed-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_fixed_env_cfg:A1AmpFixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpFixedCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Fixed-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_fixed_env_cfg:A1AmpFixedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpFixedCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Range-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_range_env_cfg:A1AmpRangeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpRangeCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Range-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_range_env_cfg:A1AmpRangeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpRangeCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Walk-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_walk_env_cfg:A1AmpWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpWalkCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Walk-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_walk_env_cfg:A1AmpWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpWalkCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-ADD-A1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAddCfg",
    },
)
