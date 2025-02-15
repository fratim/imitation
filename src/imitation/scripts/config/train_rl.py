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
def acrobot():
    common = dict(env_name="Acrobot-v1")


@train_rl_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")
    rl = dict(batch_size=16384)
    total_timesteps = int(5e6)


@train_rl_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    total_timesteps = int(1e5)


@train_rl_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1e6)


@train_rl_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v2")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving

@train_rl_ex.named_config
def seals_half_cheetah():
    common = dict(env_name="seals/HalfCheetah-v0")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving

@train_rl_ex.named_config
def seals_half_cheetah_trunc():
    common = dict(env_name="seals/HalfCheetah-v0")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@train_rl_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")

@train_rl_ex.named_config
def seals_hopper_trunc():
    common = dict(env_name="seals/Hopper-v0")


@train_rl_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")
    rl = dict(batch_size=16384)
    total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@train_rl_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")

@train_rl_ex.named_config
def mountain_car_trunc():
    common = dict(env_name="MountainCar-v0")


@train_rl_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")
    total_timesteps = int(2e5)
    rl = dict(batch_size=2048,
              rl_kwargs=
              dict(
                  learning_rate=3e-4,
                  batch_size=64,
                  n_epochs=10,
                  ent_coef=0.0,
              )
              )

@train_rl_ex.named_config
def seals_mountain_car_trunc():
    common = dict(env_name="seals/MountainCar-v0")
    total_timesteps = int(2e5)
    rl = dict(batch_size=2048,
              rl_kwargs=
              dict(
                  learning_rate=3e-4,
                  batch_size=64,
                  n_epochs=10,
                  ent_coef=0.0,
              )
              )


@train_rl_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")
    rl = dict(
        batch_size=4096,
        rl_kwargs=dict(
            gamma=0.9,
            learning_rate=1e-3,
        ),
    )
    total_timesteps = int(2e5)


@train_rl_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")


@train_rl_ex.named_config
def seals_ant():
    total_timesteps = int(1e7)
    common = dict(env_name="seals/Ant-v0")
    rl = dict(batch_size=512,
              rl_kwargs=
                dict(
                    ## Parameters specifically for ant from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
                    batch_size=32,
                    gamma=0.98,
                    learning_rate=1.90609e-05,
                    ent_coef=4.9646e-07,
                    clip_range=0.1,
                    n_epochs=10,
                    gae_lambda=0.8,
                    max_grad_norm=0.6,
                    vf_coef=0.677239
                )
    )


@train_rl_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")


@train_rl_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")

@train_rl_ex.named_config
def seals_walker_trunc():
    common = dict(env_name="seals/Walker2d-v0")


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
