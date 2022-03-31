import hydra

from imitation.util.util import hydra_to_sacred

from imitation.scripts import eval_dimensions


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    # run training of RL policy
    eval_dimensions.main_console(hydra_to_sacred(config.sacred["eval_dimensions"]))

    # evaluate RL policy and save video
    # eval_policy.main_console(hydra_to_sacred(config.sacred["eval_rl_policy"]), flag="RL")

if __name__ == "__main__":
    main()