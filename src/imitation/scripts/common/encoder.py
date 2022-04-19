"""Common configuration elements for encoder network training."""

import logging
from typing import Any, Mapping, Type, Sequence

import sacred
from stable_baselines3.common import vec_env

from imitation.rewards import reward_nets
from imitation.util import networks

from imitation.scripts.common import common as common_config

encoder_ingredient = sacred.Ingredient("encoder")
logger = logging.getLogger(__name__)


@encoder_ingredient.config
def config():
    # Custom reward network
    net_cls = reward_nets.BasicEncoderNet
    net_kwargs = {}
    locals()  # quieten flake8

def invert_tuple(input_tuple):
    inverted_tuple = input_tuple[::-1]
    return inverted_tuple

@encoder_ingredient.capture
def make_encoder_net(
    encoder_type: str,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
    output_dim: int,
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
        return reward_nets.BasicEncoder(output_dimension=output_dim, target_states=None)

    elif encoder_type == "reduction":
        target_states = common_config.get_reduced_state_space()
        assert output_dim == len(target_states)
        return reward_nets.BasicEncoder(output_dimension=output_dim, target_states=target_states)

    elif encoder_type == "reduction_inverted":
        target_states = common_config.get_reduced_state_space()
        target_states = invert_tuple(target_states)
        assert output_dim == len(target_states)
        return reward_nets.BasicEncoder(output_dimension=output_dim, target_states=target_states)

    elif encoder_type == "network":
        encoder_net = net_cls(
            venv.observation_space,
            venv.action_space,
            output_dim,
            target_states=None,
            **net_kwargs,
        )
        logging.info(f"Encoder network:\n {encoder_net}")
        return encoder_net

    elif encoder_type == "reduction_followed_by_network":
        target_states = common_config.get_reduced_state_space()
        encoder_net = net_cls(
            venv.observation_space,
            venv.action_space,
            output_dim,
            target_states=target_states,
            **net_kwargs,
        )
        logging.info(f"Encoder network:\n {encoder_net}")
        return encoder_net

    else:
        raise ValueError(f"Unknown Encoder Type: {encoder_type}")
