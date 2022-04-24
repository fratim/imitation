import hydra

from imitation.util.util import hydra_to_sacred

from imitation.scripts import train_rl, eval_policy


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    # run training of RL policy
    # train_rl.main_console(hydra_to_sacred(config.sacred["train_rl"]))

    # evaluate RL policy and save video
    eval_policy.main_console(hydra_to_sacred(config.sacred["eval_rl_policy"]), flag="RL")

if __name__ == "__main__":
    main()