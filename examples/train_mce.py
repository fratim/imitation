import matplotlib.pyplot as plt

from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
)
import gym
import imitation.envs.examples.model_envs
from imitation.algorithms import base
import numpy as np

from imitation.data import rollout
from imitation.envs import resettable_env
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.rewards import reward_nets


DEMO_STEPS = 10000
EXPERT_TRAJS = "all"
TRAIN_ITER = 1000
TARGET_STATES = (1,)


def train_mce_irl(demos, env, state_venv, **kwargs):
    reward_net = reward_nets.BasicRewardNet(
        env.pomdp_observation_space,
        env.action_space,
        use_action=False,
        use_next_state=False,
        use_done=False,
        hid_sizes=[]
    )

    mce_irl = MCEIRL(demos, env, reward_net, linf_eps=1e-3)
    mce_irl.train(max_iter=TRAIN_ITER, **kwargs)

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(DEMO_STEPS),
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

    return mce_irl, reward_net

def make_envs():
    # env_name = "imitation/CliffWorld7x4-v0"
    env_name = "imitation/GridWorld7x4-v0"
    env = gym.make(env_name)
    state_venv = resettable_env.DictExtractWrapper(
        DummyVecEnv([lambda: gym.make(env_name)] * 4), "state"
    )
    obs_venv = resettable_env.DictExtractWrapper(
        DummyVecEnv([lambda: gym.make(env_name)] * 4), "obs"
    )

    env_name_reduced = "imitation/GridWorld7x1-v0"
    env_reduced = gym.make(env_name_reduced)
    state_venv_reduced = resettable_env.DictExtractWrapper(
        DummyVecEnv([lambda: gym.make(env_name_reduced)] * 4), "state"
    )
    obs_venv_reduced = resettable_env.DictExtractWrapper(
        DummyVecEnv([lambda: gym.make(env_name_reduced)] * 4), "obs"
    )

    return env, env_reduced, state_venv, state_venv_reduced

def main():

    env, env_reduced, state_venv, state_venv_reduced = make_envs()

    _, _, pi = mce_partition_fh(env, policy_temp=7)

    _, om = mce_occupancy_measures(env, pi=pi)

    plot_om(om.reshape(env.height, env.width), "occupancy")

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

    ################ LEARN FROM OCCUPANCY MEASURE ##########################
    om = om.reshape(env.height, env.width)
    om_reduced = np.mean(om, axis=0)
    mce_irl_from_om, reward_net = train_mce_irl(om_reduced, env_reduced, state_venv_reduced)
    mce_irl_from_om.plot_reward_map(reward_net, "om")

    ################## LEARN FROM TRAJECTORIES ##############################
    if EXPERT_TRAJS != "all":
        expert_trajs_passed = expert_trajs[0:EXPERT_TRAJS]
    else:
        expert_trajs_passed = expert_trajs

    for traj in expert_trajs_passed:
        traj.obs = [elem % env.width for elem in traj.obs]

    mce_irl_from_trajs, reward_net = train_mce_irl(expert_trajs_passed, env_reduced, state_venv_reduced)
    mce_irl_from_trajs.plot_reward_map(reward_net, "trajs")

def plot_om(om, id):
    plt.imshow(om)
    plt.savefig(f"om_{id}.png")

if __name__ == "__main__":
    main()

