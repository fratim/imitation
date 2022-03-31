"""Train GAIL or AIRL."""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Type

import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations, reward, rl, train
from imitation.scripts.config.train_adversarial import train_adversarial_ex

logger = logging.getLogger("imitation.scripts.eval_dimensions")


def _add_hook(ingredient: sacred.Ingredient) -> None:
    # This is an ugly hack around Sacred config brokenness.
    # Config hooks only apply to their current ingredient,
    # and cannot update things in nested ingredients.
    # So we have to apply this hook to every ingredient we use.
    @ingredient.config_hook
    def hook(config, command_name, logger):
        del logger
        path = ingredient.path
        if path == "train_adversarial":
            path = ""
        ingredient_config = sacred.utils.get_by_dotted_path(config, path)
        return ingredient_config["algorithm_specific"].get(command_name, {})

    # We add this so Sacred doesn't complain that algorithm_specific is unused
    @ingredient.capture
    def dummy_no_op(algorithm_specific):
        pass

    # But Sacred may then complain it isn't defined in config! So, define it.
    @ingredient.config
    def dummy_config():
        algorithm_specific = {}  # noqa: F841


for ingredient in [train_adversarial_ex] + train_adversarial_ex.ingredients:
    _add_hook(ingredient)


@train_adversarial_ex.capture
def train_adversarial(
    _run,
    _seed: int,
    show_config: bool,
    algo_cls: Type[common.AdversarialTrainer],
    algorithm_kwargs: Mapping[str, Any],
    total_timesteps: int,
    checkpoint_interval: int,
) -> Mapping[str, Mapping[str, float]]:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - DiscrimNets are saved to `f"{log_dir}/checkpoints/{step}/discrim/"`,
            where step is either the training round or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        _seed: Random seed.
        show_config: Print the merged config before starting training. This is
            analogous to the print_config command, but will show config after
            rather than before merging `algorithm_specific` arguments.
        algo_cls: The adversarial imitation learning algorithm to use.
        algorithm_kwargs: Keyword arguments for the `GAIL` or `AIRL` constructor.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` rounds and after training is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations.
    """
    if show_config:
        # Running `train_adversarial print_config` will show unmerged config.
        # So, support showing merged config from `train_adversarial {airl,gail}`.
        sacred.commands.print_config(_run)

    custom_logger, log_dir = common_config.setup_logging()
    expert_trajs = demonstrations.load_expert_trajs()



def main_console(parameters=None):
    observer = FileStorageObserver(osp.join("output", "sacred", "train_adversarial"))
    train_adversarial_ex.observers.append(observer)
    if parameters is None:
        train_adversarial_ex.run_commandline()
    elif isinstance(parameters, str):
        train_adversarial_ex.run_commandline(parameters)
    else:
        train_adversarial_ex.run(command_name=parameters["command_name"], config_updates=parameters["config_updates"], named_configs=parameters["named_configs"])

if __name__ == "__main__":  # pragma: no cover
    main_console()
