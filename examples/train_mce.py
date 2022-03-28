
from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
)
import gym
import imitation.envs.examples.model_envs
from imitation.algorithms import base

from imitation.data import rollout
from imitation.envs import resettable_env
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.rewards import reward_nets


env_name = "imitation/CliffWorld15x6-v0"
env = gym.make(env_name)
state_venv = resettable_env.DictExtractWrapper(
    DummyVecEnv([lambda: gym.make(env_name)] * 4), "state"
)
obs_venv = resettable_env.DictExtractWrapper(
    DummyVecEnv([lambda: gym.make(env_name)] * 4), "obs"
)
_, _, pi = mce_partition_fh(env)

_, om = mce_occupancy_measures(env, pi=pi)

expert = TabularPolicy(
    state_space=env.pomdp_state_space,
    action_space=env.action_space,
    pi=pi,
    rng=None,
)

expert_trajs = rollout.generate_trajectories(
    policy=expert,
    venv=state_venv,
    sample_until=rollout.make_min_timesteps(5000),
)

print("Expert stats: ", rollout.rollout_stats(expert_trajs))

def train_mce_irl(demos, **kwargs):
    reward_net = reward_nets.BasicRewardNet(
        env.pomdp_observation_space,
        env.action_space,
        use_action=False,
        use_next_state=False,
        use_done=False,
        hid_sizes=[],
    )

    mce_irl = MCEIRL(demos, env, reward_net, linf_eps=1e-3)
    mce_irl.train(**kwargs)

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

    return mce_irl

mce_irl_from_om = train_mce_irl(om)
mce_irl_from_trajs = train_mce_irl(expert_trajs[0:10])
