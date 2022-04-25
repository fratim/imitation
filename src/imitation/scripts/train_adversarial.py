"""Train GAIL or AIRL."""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Type

import sacred.commands
import torch as th
import torch.jit
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.common import demonstrations, reward, rl, train, encoder
from imitation.scripts.config.train_adversarial import train_adversarial_ex

from imitation.scripts.common import common as common_config

logger = logging.getLogger("imitation.scripts.train_adversarial")


def init_networks(load_path, venv, expert_trajs, encoder_learner_kwargs):
    if load_path is None:
        gen_algo = rl.make_rl_algo(venv)
        encoder_net_expert = encoder.make_encoder_net(**demonstrations.get_encoder_kwargs(expert_trajs=expert_trajs))
        encoder_net_learner = encoder.make_encoder_net(venv=venv, output_dim=encoder_net_expert.output_dimension,
                                                       **encoder_learner_kwargs)
        reward_net = reward.make_reward_net(input_dimension=encoder_net_expert.output_dimension * 2)
    else:

        gen_algo = rl.load_rl_algo(load_path=os.path.join(load_path, "gen_policy", "model.zip"))

        assert "network" not in demonstrations.get_encoder_kwargs(expert_trajs=expert_trajs)["encoder_type"]
        encoder_net_expert = encoder.make_encoder_net(**demonstrations.get_encoder_kwargs(expert_trajs=expert_trajs))

        if "network" in encoder_learner_kwargs["encoder_type"]:
            encoder_net_learner = torch.jit.load(os.path.join(load_path, "encoder_net.pt"))
        else:
            encoder_net_learner = encoder.make_encoder_net(venv=venv, output_dim=encoder_net_expert.output_dimension,
                                                           **encoder_learner_kwargs)

        reward_net = torch.load(os.path.join(load_path, "reward_train.pt"))
        reward_net = reward_net.base

    return gen_algo, encoder_net_expert, encoder_net_learner, reward_net

def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer._reward_net, os.path.join(save_path, "reward_net.pt"))
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
    )
    if trainer._encoder_net.type == "network":
        th.save(trainer._encoder_net, os.path.join(save_path, "encoder_net.pt"))
    if trainer._encoder_net_expert.type == "network":
        th.save(trainer._encoder_net_expert, os.path.join(save_path, "encoder_net_expert.pt"))

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
    encoder_learner_kwargs,
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

    venv = common_config.make_venv()
    eval_env = common_config.make_venv(num_vec=1, parallel=False)

    # load_path = "/work/frtim/airl_runs/red_basev5/2022-04-23_14-31-03/dem=seals_hopper_seed_three,env=seals_hopper_org,exp_enc=reduced,learner_enc=reduced,n_disc=1000,rl_lr=0.001,seed=six,task_name=red_basev5,wandb=yes/logss/checkpoints/00900"
    load_path = None

    gen_algo, encoder_net_expert, encoder_net_learner, reward_net = init_networks(load_path, venv, expert_trajs, encoder_learner_kwargs)

    logger.info(f"Using '{algo_cls}' algorithm")
    algorithm_kwargs = dict(algorithm_kwargs)
    for k in ("shared", "airl", "gail"):
        # Config hook has copied relevant subset of config to top-level.
        # But due to Sacred limitations, cannot delete the rest of it.
        # So do that here to avoid passing in invalid arguments to constructor.
        if k in algorithm_kwargs:
            del algorithm_kwargs[k]


    trainer = algo_cls(
        venv=venv,
        demonstrations=expert_trajs,
        gen_algo=gen_algo,
        log_dir=log_dir,
        reward_net=reward_net,
        custom_logger=custom_logger,
        encoder_net=encoder_net_learner,
        encoder_net_expert=encoder_net_expert,
        eval_env=eval_env,
        **algorithm_kwargs,
    )

    def callback(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))

    trainer.train(total_timesteps, callback)

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    return_obj = {
        "imit_stats": train.eval_policy(trainer.policy, trainer.venv_train),
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }

    relative_mean_reward_imitator = return_obj["imit_stats"]["return_mean"] / return_obj["expert_stats"]["return_mean"]

    return relative_mean_reward_imitator




@train_adversarial_ex.command
def gail():
    return train_adversarial(algo_cls=gail_algo.GAIL)


@train_adversarial_ex.command
def airl():
    return train_adversarial(algo_cls=airl_algo.AIRL)


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
