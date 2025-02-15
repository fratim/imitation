"""Configuration for imitation.scripts.train_adversarial."""

import sacred

from imitation.rewards import reward_nets
from imitation.scripts.common import common, demonstrations, reward, rl, train

train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        train.train_ingredient,
    ],
)

# trunc environments: mountain_car_trunc,seals_half_cheetah_trunc,seals_hopper_trunc,seals_mountain_car_trunc,seals_walker_trunc


@train_adversarial_ex.config
def defaults():
    show_config = False

    total_timesteps = int(5e5)  # Num of environment transitions to sample
    algorithm_kwargs = dict(
        demo_batch_size=1024,  # Number of expert samples per discriminator update
        n_disc_updates_per_round=4,  # Num discriminator updates per generator round
        gen_replay_buffer_capacity=2048,
    )
    algorithm_specific = {}  # algorithm_specific[algorithm] is merged with config

    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)


@train_adversarial_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, rl):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.
    algorithm_kwargs["gen_replay_buffer_capacity"] = rl["batch_size"]


# Shared settings

MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))

ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    algorithm_kwargs=dict(shared=dict(demo_batch_size=8192)),
    rl=dict(batch_size=16384),
)


# Classic RL Gym environment named configs


@train_adversarial_ex.named_config
def acrobot():
    env_name = "Acrobot-v1"
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1.4e6)


@train_adversarial_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")
    algorithm_kwargs = {"allow_variable_horizon": True}

@train_adversarial_ex.named_config
def mountain_car_trunc():
    common = dict(env_name="MountainCar-v0")
    algorithm_kwargs = {"allow_variable_horizon": True}
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNetTruncated,
                net_kwargs=dict(
                    target_states=tuple((0,)),
                )
            ),
        ),
    )


@train_adversarial_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_adversarial_ex.named_config
def seals_mountain_car_trunc():
    common = dict(env_name="seals/MountainCar-v0")
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNetTruncated,
                net_kwargs=dict(
                    target_states=tuple((0,)),
                )
            ),
            gail=dict(
                net_cls=reward_nets.BasicRewardNetTruncated,
                net_kwargs=dict(
                    target_states=tuple((0,)),
                )
            ),
        ),
    )


@train_adversarial_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")



# Standard MuJoCo Gym environment named configs


@train_adversarial_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    common = dict(env_name="seals/Ant-v0")


@train_adversarial_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="HalfCheetah-v2")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )

@train_adversarial_ex.named_config
def seals_half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/HalfCheetah-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )

@train_adversarial_ex.named_config
def seals_half_cheetah_trunc():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/HalfCheetah-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNetTruncated,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                    target_states=tuple((0, 1, 2)),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )


@train_adversarial_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Hopper-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )

@train_adversarial_ex.named_config
def seals_hopper_trunc():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Hopper-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNetTruncated,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                    target_states=tuple((0, 1, 2)),
                ),
            ),
            gail=dict(
                net_cls=reward_nets.BasicRewardNetTruncated,
                net_kwargs=dict(
                    target_states=tuple((0, 1, 2)),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )


@train_adversarial_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_adversarial_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_swimmer():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Swimmer-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )


@train_adversarial_ex.named_config
def seals_walker():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Walker2d-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )

@train_adversarial_ex.named_config
def seals_walker_trunc():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Walker2d-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNetTruncated,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                    target_states=tuple((0, 1, 2))
                ),
            ),
            gail=dict(
                net_cls=reward_nets.BasicRewardNetTruncated,
                net_kwargs=dict(
                    target_states=tuple((0, 1, 2)),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )

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
    )
