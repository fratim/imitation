"""Configuration for imitation.scripts.train_adversarial."""

import sacred

from imitation.rewards import reward_nets
from imitation.scripts.common import common, demonstrations, reward, rl, train, encoder
from imitation.util import networks

train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        train.train_ingredient,
        encoder.encoder_ingredient
    ],
)

# trunc environments: mountain_car_trunc,seals_half_cheetah_trunc,seals_hopper_trunc,seals_mountain_car_trunc,seals_walker_trunc


@train_adversarial_ex.config
def defaults():
    show_config = False

    total_timesteps = int(1e7)# Num of environment transitions to sample

    algorithm_kwargs = dict(
        demo_batch_size=100,  # Number of expert samples per discriminator update
        n_gen_updates_per_round=1000,
        n_disc_updates_per_round=1000,  # Num discriminator updates per generator round
        n_enc_updates_per_round=1000,
        disc_lr=1e-3,
        enc_lr=1e-3,
    )

    checkpoint_interval = 50  # Num epochs between checkpoints (<0 disables)

@train_adversarial_ex.named_config
def identity():
    encoder_learner_kwargs = dict(
        encoder_type="identity",
    )

@train_adversarial_ex.named_config
def reduced():
    encoder_learner_kwargs = dict(
        encoder_type="reduction",
    )

@train_adversarial_ex.named_config
def network():
    encoder_learner_kwargs = dict(
        encoder_type="network",
    )

@train_adversarial_ex.named_config
def reduced_network():
    encoder_learner_kwargs = dict(
        encoder_type="reduction_followed_by_network",
    )


@train_adversarial_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")

@train_adversarial_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")

@train_adversarial_ex.named_config
def seals_half_cheetah():
    common = dict(env_name="seals/HalfCheetah-v0")

@train_adversarial_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")

@train_adversarial_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")

# Debug configs

@train_adversarial_ex.named_config
def test():
    total_timesteps = 30000

@train_adversarial_ex.named_config
def fast():
    # Minimize the amount of computation. Useful for test cases.

    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    algorithm_kwargs = dict(
        demo_batch_size=1,
        n_disc_updates_per_round=4,
        disc_lr=1e-3,
        n_enc_updates_per_round=4,
    )
