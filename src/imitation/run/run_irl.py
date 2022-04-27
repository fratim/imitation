import hydra
import imitation

from imitation.util.util import hydra_to_sacred

from imitation.scripts import train_adversarial, eval_policy

import wandb
import omegaconf
import numpy
from git import Repo

from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):

    repo = Repo(imitation.__file__, search_parent_directories=True)
    t = repo.head.commit.tree
    diff_message = repo.git.diff(t)

    f = open("git_diff.txt", "w")
    f.write(diff_message)
    f.close()

    wandb.config = omegaconf.OmegaConf.to_container(
        config.sacred["train_adversarial"], resolve=True, throw_on_missing=True
    )

    # run training of IRL policy
    relative_mean_reward_imitator = train_adversarial.main_console(hydra_to_sacred(config.sacred["train_adversarial"]))

    # evaluate RL policy and save video
    eval_policy.main_console(hydra_to_sacred(config.sacred["eval_irl_policy"]), flag="IRL")

    return relative_mean_reward_imitator

if __name__ == "__main__":
    main()