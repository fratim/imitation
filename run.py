import hydra
from omegaconf import OmegaConf


from src.imitation.scripts import train_rl
from src.imitation.scripts import train_adversarial
from src.imitation.scripts import eval_policy

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    # run training of RL policy
    train_rl.main_console(hydra_to_sacred(config.sacred["train_rl"]))

    # evaluate RL policy and save video
    # eval_policy.main_console(hydra_to_sacred(config.sacred["eval_rl_policy"]), flag="RL")

    # run training of IRL policy
    train_adversarial.main_console(hydra_to_sacred(config.sacred["train_adversarial"]))

    # evaluate RL policy and save video
    eval_policy.main_console(hydra_to_sacred(config.sacred["eval_irl_policy"]), flag="IRL")

def hydra_to_sacred(input):
    input_transformed = dict()
    input_transformed["named_configs"] = OmegaConf.to_object(input["named_configs"])
    input_transformed["config_updates"] = OmegaConf.to_object(input["config_updates"])

    if "command_name" in input.keys():
        input_transformed["command_name"] = str(input["command_name"])

    return input_transformed


if __name__ == "__main__":
    main()
