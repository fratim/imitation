import hydra

from run_rl import main as run_rl_main
from run_irl import main as run_irl_main


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config):

    run_rl_main(config)
    run_irl_main(config)


if __name__ == "__main__":
    main()
