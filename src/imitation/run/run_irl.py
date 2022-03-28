import hydra

from imitation.util.util import hydra_to_sacred

from imitation.scripts import train_adversarial, eval_policy


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):

    # run training of IRL policy
    train_adversarial.main_console(hydra_to_sacred(config.sacred["train_adversarial"]))

    # evaluate RL policy and save video
    eval_policy.main_console(hydra_to_sacred(config.sacred["eval_irl_policy"]), flag="IRL")

if __name__ == "__main__":
    main()