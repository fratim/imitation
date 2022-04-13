"""Common configuration element for scripts learning from demonstrations."""

import logging
import os
from typing import Optional, Sequence

import sacred

from imitation.data import types
import copy

demonstrations_ingredient = sacred.Ingredient("demonstrations")
logger = logging.getLogger(__name__)


@demonstrations_ingredient.config
def config():
    # Demonstrations
    data_dir = "data/"
    rollout_path = None  # path to file containing rollouts
    n_expert_demos = None  # Num demos used. None uses every demo possible
    encoder_kwargs = dict(
        encoder_type="identity",
        target_states=None
    )
    locals()  # quieten flake8

@demonstrations_ingredient.named_config
def identity():
    encoder_kwargs = dict(
        encoder_type="identity",
        target_states=None
    )

@demonstrations_ingredient.named_config
def reduced():
    encoder_kwargs = dict(
        encoder_type="reduction",
        target_states=(0, 1, 2)
    )

@demonstrations_ingredient.named_config
def reduced_inverted():
    encoder_kwargs = dict(
        encoder_type="reduction",
        target_states=invert_tuple((0, 1, 2))
    )

@demonstrations_ingredient.named_config
def fast():
    n_expert_demos = 1  # noqa: F841

@demonstrations_ingredient.capture
def get_output_dim(expert_trajs, encoder_kwargs) -> int:
    if encoder_kwargs["target_states"] is not None:
        return len(encoder_kwargs["target_states"])
    else:
        return expert_trajs[0].obs.shape[1]

@demonstrations_ingredient.capture
def guess_expert_dir(data_dir: str, env_name: str) -> str:
    rollout_hint = env_name.rsplit("-", 1)[0].replace("/", "_").lower()
    return os.path.join(data_dir, "expert_models", f"{rollout_hint}_0")

@demonstrations_ingredient.capture
def get_encoder_kwargs(expert_trajs, encoder_kwargs):
    encoder_kwargs_full = copy.deepcopy(encoder_kwargs)
    encoder_kwargs_full["output_dim"] = get_output_dim(expert_trajs=expert_trajs)
    return encoder_kwargs_full

def invert_tuple(input_tuple):
    inverted_tuple = input_tuple[::-1]
    return inverted_tuple


@demonstrations_ingredient.config_hook
def hook(config, command_name, logger):
    """If rollout_path not set explicitly, then guess it based on environment name."""
    del command_name, logger
    updates = {}
    if config["demonstrations"]["rollout_path"] is None:
        data_dir = config["demonstrations"]["data_dir"]
        env_name = config["common"]["env_name"].replace("/", "_")
        updates["rollout_path"] = os.path.join(
            guess_expert_dir(data_dir, env_name),
            "rollouts",
            "final.pkl",
        )
    return updates


@demonstrations_ingredient.capture
def load_expert_trajs(
    rollout_path: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = types.load(rollout_path)
    logger.info(f"Loaded {len(expert_trajs)} expert trajectories from '{rollout_path}'")
    if n_expert_demos is not None:
        if len(expert_trajs) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available via {rollout_path}.",
            )
        expert_trajs = expert_trajs[:n_expert_demos]
        logger.info(f"Truncated to {n_expert_demos} expert trajectories")
    return expert_trajs
