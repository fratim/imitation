"""Common configuration elements for encoder network training."""

import logging
from typing import Any, Mapping, Type

import sacred
from stable_baselines3.common import vec_env

from imitation.rewards import reward_nets
from imitation.util import networks

encoder_ingredient = sacred.Ingredient("encoder")
logger = logging.getLogger(__name__)


@encoder_ingredient.config
def config():
    # Custom reward network
    use_encoder = False
    net_cls = reward_nets.BasicEncoderNet
    net_kwargs = {}
    locals()  # quieten flake8

@encoder_ingredient.capture
def make_encoder_net(
    venv: vec_env.VecEnv,
    use_encoder: False,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
) -> reward_nets.RewardNet:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.

    Returns:
        None if `reward_net_cls` is None; otherwise, an instance of `reward_net_cls`.
    """

    if use_encoder == False:
        return None
    else:
        encoder_net = net_cls(
            venv.observation_space,
            venv.action_space,
            **net_kwargs,
        )
        logging.info(f"Encoder network:\n {encoder_net}")
        return encoder_net
