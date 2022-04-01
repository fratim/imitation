"""Train GAIL or AIRL."""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Type

import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations, reward, rl, train
from imitation.scripts.config.train_adversarial import train_adversarial_ex

import sacred
from imitation.scripts.config.eval_dimensions import eval_dimensions_ex



logger = logging.getLogger("imitation.scripts.eval_dimensions")


@eval_dimensions_ex.main
def eval_dimensions(
    _run,
    _seed: int,
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
    # Running `train_adversarial print_config` will show unmerged config.
    # So, support showing merged config from `train_adversarial {airl,gail}`.
    sacred.commands.print_config(_run)

    custom_logger, log_dir = common_config.setup_logging()
    expert_trajs_good = demonstrations.load_expert_trajs(specification="good")
    expert_trajs_bad = demonstrations.load_expert_trajs(specification="bad")

    get_relevant_dimensions(expert_trajs_good, expert_trajs_bad)

    print("done")

def add_trajectories(trajs, X, y, y_label):
    for traj in trajs:
        for j in range(len(traj.obs) - 1):
            entry = []
            for elem1, elem2 in zip(traj.obs[j], traj.obs[j+1]):
                entry.append(elem1)
                entry.append(elem2)

            X.append(entry)
            y.append([y_label])

    return X, y

def get_relevant_dimensions(good_trajs, bad_trajs):

    X = list()
    y = list()

    X, y = add_trajectories(good_trajs, X, y, y_label=1)
    X, y = add_trajectories(bad_trajs, X, y, y_label=0)

    X = np.array(X)
    y = np.array(y)[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # both row and col coordinate
    clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1", C=0.05).fit(X_train, y_train)
    print(f"Test classification score both: {clf.score(X_test, y_test)}")

    cs = []
    accuracies = []
    n_components = []

    for c in range(1000000, 1, -100):
        c_used = 0.1/c
        print(f"##################################")
        print(f"c used: {c_used}")
        clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1", C=c_used).fit(X_train, y_train)
        coef = clf.coef_
        print(coef)
        n_components.append(np.count_nonzero(coef))
        accuracy = clf.score(X_test, y_test)
        print(f"Test classification score: {accuracy}")
        cs.append(c_used)
        accuracies.append(accuracy)

    plt.plot(cs, accuracies, cs, n_components)
    plt.show()



def main_console(parameters=None):
    observer = FileStorageObserver(osp.join("output", "sacred", "eval_dimensions"))
    eval_dimensions_ex.observers.append(observer)
    if parameters is None:
        eval_dimensions_ex.run_commandline()
    elif isinstance(parameters, str):
        eval_dimensions_ex.run_commandline(parameters)
    else:
        eval_dimensions_ex.run(config_updates=parameters["config_updates"], named_configs=parameters["named_configs"])

if __name__ == "__main__":  # pragma: no cover
    main_console()
