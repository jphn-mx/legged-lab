import gymnasium as gym

from legged_lab.envs import ManagerBasedAmpEnv

from . import agents

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Benji-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Benji-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-ADD-A1-Benji-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_env_cfg:A1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAddCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Benji-Range-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_range_env_cfg:A1AmpRangeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpRangeCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-A1-Benji-Range-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_amp_range_env_cfg:A1AmpRangeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1RslRlOnPolicyRunnerAmpRangeCfg",
    },
)