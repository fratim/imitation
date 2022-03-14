import hydra

from src.imitation.scripts import train_rl
from src.imitation.scripts import train_adversarial


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    print(f"dummy config is: {config.dummy_config}")
    train_rl.main_console(["train_rl"] + config.sacred.train_rl_command.split())

    print(f"dummy config is: {config.dummy_config}")
    train_adversarial.main_console(["train_adversarial"] + config.sacred.train_adversarial_command.split())

    print(f"dummy config is: {config.dummy_config}")


if __name__ == "__main__":
    main()