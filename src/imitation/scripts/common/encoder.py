"""Common configuration elements for encoder network training."""

import logging
from typing import Any, Mapping, Type, Sequence

import sacred
from stable_baselines3.common import vec_env

from imitation.rewards import reward_nets
from imitation.util import networks

encoder_ingredient = sacred.Ingredient("encoder")
logger = logging.getLogger(__name__)


@encoder_ingredient.config
def config():
    # Custom reward network
    net_cls = reward_nets.BasicEncoderNet
    net_kwargs = {}
    locals()  # quieten flake8

@encoder_ingredient.capture
def make_encoder_net(
    encoder_type: str,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
    output_dim: int,
    target_states: Sequence[int] = None,
    venv: vec_env.VecEnv = None,
) -> reward_nets.RewardNet:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.

    Returns:
        None if `reward_net_cls` is None; otherwise, an instance of `reward_net_cls`.
    """

    if encoder_type == "identity":
        return reward_nets.BasicEncoder(target_states=None, output_dimension=output_dim)

    elif encoder_type == "reduction":
        assert venv is None
        assert output_dim == len(target_states)
        return reward_nets.BasicEncoder(target_states=target_states, output_dimension=output_dim)

    elif encoder_type == "network":
        encoder_net = net_cls(
            venv.observation_space,
            venv.action_space,
            output_dim,
            **net_kwargs,
        )
        logging.info(f"Encoder network:\n {encoder_net}")
        return encoder_net

    elif encoder_type == "reduction_followed_by_network":
        encoder_net = net_cls(
            venv.observation_space,
            venv.action_space,
            output_dim,
            **net_kwargs,
        )
        logging.info(f"Encoder network:\n {encoder_net}")
        return encoder_net

    else:
        raise ValueError(f"Unknown Encoder Type: {encoder_type}")
