"""Configuration settings for train_rl, training a policy with RL."""

import sacred

from imitation.scripts.common import common, rl, train

train_rl_ex = sacred.Experiment(
    "train_rl",
    ingredients=[common.common_ingredient, train.train_ingredient, rl.rl_ingredient],
)


@train_rl_ex.config
def train_rl_defaults():
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    normalize_reward = True  # Use VecNormalize to normalize the reward
    normalize_kwargs = dict()  # kwargs for `VecNormalize`

    # If specified, overrides the ground-truth environment reward
    reward_type = None  # override reward type
    reward_path = None  # override reward path

    rollout_save_final = True  # If True, save after training is finished.
    rollout_save_n_timesteps = None  # Min timesteps saved per file, optional.
    rollout_save_n_episodes = None  # Num episodes saved per file, optional.

    policy_save_interval = 10000  # Num timesteps between saves (<=0 disables)
    policy_save_final = True  # If True, save after training is finished.


@train_rl_ex.config
def default_end_cond(rollout_save_n_timesteps, rollout_save_n_episodes):
    # Only set default if both end cond options are None.
    # This way the Sacred CLI caller can set `rollout_save_n_episodes` only
    # without getting an error that `rollout_save_n_timesteps is not None`.
    if rollout_save_n_timesteps is None and rollout_save_n_episodes is None:
        rollout_save_n_timesteps = 20000  # Min timesteps saved per file, optional.


# Standard Gym env configs
@train_rl_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")

@train_rl_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")

@train_rl_ex.named_config
def seals_half_cheetah():
    common = dict(env_name="seals/HalfCheetah-v0")

@train_rl_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")

@train_rl_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")


# Debug configs
@train_rl_ex.named_config
def test():
    """Intended for testing purposes: small # of updates, ends quickly."""
    total_timesteps = int(20000)


@train_rl_ex.named_config
def fast():
    """Intended for testing purposes: small # of updates, ends quickly."""
    total_timesteps = int(4)
    policy_save_interval = 2
