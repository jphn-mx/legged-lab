import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="LeggedLab-Isaac-Velocity-Flat-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:A1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1FlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-Velocity-Flat-A1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:A1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1FlatPPORunnerCfg",
    },
)
