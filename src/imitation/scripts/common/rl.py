"""Common configuration elements for reinforcement learning."""

import logging
from typing import Any, Mapping, Type

import sacred
import stable_baselines3
from stable_baselines3.common import (
    base_class,
    off_policy_algorithm,
    on_policy_algorithm,
    vec_env,
)

from stable_baselines3.common.noise import NormalActionNoise

from imitation.scripts.common.train import train_ingredient
from stable_baselines3.common.buffers import ReplayBuffer

rl_ingredient = sacred.Ingredient("rl", ingredients=[train_ingredient])
logger = logging.getLogger(__name__)


@rl_ingredient.config
def config():
    rl_cls = stable_baselines3.TD3
    batch_size = 2048  # batch size for RL algorithm

    rl_kwargs = dict(
        # parameters taken from DA paper
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4, # TODO-tim implement learning rate scheduling
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        # unverified parameters
        policy="MlpPolicy",
        ## Actor Network
        # should be: dense(400), relu, dense(300), relu, dense(actiondim), "tanh"
        # initialized as described in common.actor
        ## Critic should be Dense(400), tanh, Dense(300), tanh, (Dense)
        learning_starts=1000, # 1000 in DAC, 10000 normally for TD3
        batch_size=100,
        # Buffer with infinite size
        replay_buffer_class=ReplayBuffer,
        buffer_size=100000000,
        optimize_memory_usage=False,
        policy_delay=1,       # unclear stuff from D3 hopper baseline implementation
        action_noise=NormalActionNoise(mean=[0, 0, 0], sigma=[0.1, 0.1, 0.1])) # TODO-tim what is action noise used for? how is this set in the original implementation?
    locals()  # quieten flake8

# @rl_ingredient.config
# def config():
#     rl_cls = stable_baselines3.PPO
#     batch_size = 2048  # batch size for RL algorithm
#     rl_kwargs = dict(
#         # For recommended PPO hyperparams in each environment, see:
#         # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
#         learning_rate=3e-4,
#         batch_size=64,
#         n_epochs=10,
#         ent_coef=0.0,
#     )
#     locals()  # quieten flake8

@rl_ingredient.named_config
def fast():
    batch_size = 2
    # SB3 RL seems to need batch size of 2, otherwise it runs into numeric
    # issues when computing multinomial distribution during predict()
    rl_kwargs = dict(batch_size=2)
    locals()  # quieten flake8


@rl_ingredient.capture
def make_rl_algo(
    venv: vec_env.VecEnv,
    rl_cls: Type[base_class.BaseAlgorithm],
    batch_size: int,
    rl_kwargs: Mapping[str, Any],
    train: Mapping[str, Any],
    _seed: int,
) -> base_class.BaseAlgorithm:
    """Instantiates a Stable Baselines3 RL algorithm.

    Args:
        venv: The vectorized environment to train on.
        rl_cls: Type of a Stable Baselines3 RL algorithm.
        batch_size: The batch size of the RL algorithm.
        rl_kwargs: Keyword arguments for RL algorithm constructor.
        train: Configuration for the train ingredient. We need the
            policy_cls and policy_kwargs component.

    Returns:
        The RL algorithm.

    Raises:
        ValueError: `gen_batch_size` not divisible by `venv.num_envs`.
        TypeError: `rl_cls` is neither `OnPolicyAlgorithm` nor `OffPolicyAlgorithm`.
    """
    if batch_size % venv.num_envs != 0:
        raise ValueError(
            f"num_envs={venv.num_envs} must evenly divide batch_size={batch_size}.",
        )
    rl_kwargs = dict(rl_kwargs)
    # If on-policy, collect `batch_size` many timesteps each update.
    # If off-policy, train on `batch_size` many timesteps each update.
    # These are different notion of batches, but this seems the closest
    # possible translation, and I would expect the appropriate hyperparameter
    # to be similar between them.
    # if issubclass(rl_cls, on_policy_algorithm.OnPolicyAlgorithm):
    #     assert (
    #         "n_steps" not in rl_kwargs
    #     ), "set 'n_steps' at top-level using 'batch_size'"
    #     rl_kwargs["n_steps"] = batch_size // venv.num_envs
    # elif issubclass(rl_cls, off_policy_algorithm.OffPolicyAlgorithm):
    #     assert "batch_size" not in rl_kwargs, "set 'batch_size' at top-level"
    #     rl_kwargs["batch_size"] = batch_size
    # else:
    #     raise TypeError(f"Unsupported RL algorithm '{rl_cls}'")


    rl_algo = rl_cls(
        # policy=train["policy_cls"],
        # policy_kwargs=train["policy_kwargs"],
        env=venv,
        seed=_seed,
        train_freq=tuple((1, "episode")),
        gradient_steps=-1,
        **rl_kwargs,
    )
    logger.info(f"RL algorithm: {type(rl_algo)}")
    logger.info(f"Policy network summary:\n {rl_algo.policy}")
    return rl_algo
