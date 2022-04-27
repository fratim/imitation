"""Configuration settings for eval_policy, evaluating pre-trained policies."""

import sacred

from imitation.scripts.common import common

eval_policy_ex = sacred.Experiment(
    "eval_policy",
    ingredients=[common.common_ingredient],
)


@eval_policy_ex.config
def replay_defaults():
    eval_n_timesteps = None  # Min timesteps to evaluate, optional.
    eval_n_episodes = 100  #set this to 100 to generate rollouts 3um episodes to evaluate, optional.

    videos = True  # save video files
    video_kwargs = {}  # argumeents to VideoWrapper
    render = False  # render to screen
    render_fps = 60  # -1 to render at full speed

    policy_type = None  # class to load policy, see imitation.policies.loader
    policy_path = None  # path to serialized policy

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load

    rollout_save_path = None  # where to save rollouts to -- if None, do not save


@eval_policy_ex.named_config
def render():
    common = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def acrobot():
    common = dict(env_name="Acrobot-v1", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def ant():
    common = dict(env_name="Ant-v2", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v2", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def seals_half_cheetah():
    common = dict(env_name="seals/HalfCheetah-v0", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def half_cheetah_trunc():
    common = dict(env_name="HalfCheetah-v2", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def seals_hopper_trunc():
    common = dict(env_name="seals/Hopper-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def mountain_car_trunc():
    common = dict(env_name="MountainCar-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_mountain_car():
    eval_n_timesteps = int(200)
    common = dict(env_name="seals/MountainCar-v0", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def seals_mountain_car_trunc():
    eval_n_timesteps = int(200)
    common = dict(env_name="seals/MountainCar-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0", num_vec=1, parallel=False)


@eval_policy_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def seals_walker_trunc():
    common = dict(env_name="seals/Walker2d-v0", num_vec=1, parallel=False)

@eval_policy_ex.named_config
def fast():
    common = dict(env_name="CartPole-v1", num_vec=1, parallel=False)
    render = True
    policy_type = "ppo"
    policy_path = "tests/testdata/expert_models/cartpole_0/policies/final/"
    eval_n_timesteps = 1
    eval_n_episodes = None
