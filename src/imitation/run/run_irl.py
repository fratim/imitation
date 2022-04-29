import hydra
import imitation
from imitation.util.util import hydra_to_sacred
from imitation.scripts import train_adversarial
import wandb
from git import Repo
from multiprocessing import Process, Queue
import copy
import os
import time
import random


def save_git_status():
    repo = Repo(imitation.__file__, search_parent_directories=True)
    t = repo.head.commit.tree
    diff_message = repo.git.diff(t)

    f = open("git_diff.txt", "w")
    f.write(diff_message)
    f.close()

def my_func(irl_config):

    time.sleep(random.random()*100) # sleep for some time between 0 and 100 seconds to avoid all processes starting at once

    os.makedirs(irl_config["working_dir"])
    os.chdir(irl_config["working_dir"])
    wandb.config = irl_config

    # run training of IRL policy
    relative_mean_reward_imitator = train_adversarial.main_console(irl_config)
    return relative_mean_reward_imitator

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):

    save_git_status()

    irl_config = hydra_to_sacred(config.sacred["train_adversarial"])

    cwd = os.getcwd()

    irl_configs = []

    for seed in range(4):
        irl_config_seedx = copy.deepcopy(irl_config)
        irl_config_seedx["config_updates"]["seed"] = seed
        irl_config_seedx["working_dir"] = os.path.join(cwd, f"seed_{seed}")
        irl_configs.append(irl_config_seedx)

    queue = Queue()
    processes = [Process(target=my_func, args=(specconfig,)) for specconfig in irl_configs]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    return None

if __name__ == "__main__":
    main()