import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
LOG_INTERVAL = None


def train_mce_irl(demos, env, state_venv, **kwargs):
    reward_net = reward_nets.BasicRewardNet(
        env.pomdp_observation_space,
        env.action_space,
        use_action=False,
        use_next_state=False,
        use_done=False,
        hid_sizes=[]
    )

    mce_irl = MCEIRL(demos, env, reward_net, linf_eps=1e-3, log_interval=LOG_INTERVAL)
    mce_irl.train(max_iter=TRAIN_ITER, **kwargs)

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(DEMO_STEPS),
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

    return mce_irl, reward_net

def get_envs(identifier):
    env_name = f"imitation/GridWorld7x4x{identifier}-v0"
    env = gym.make(env_name)
    state_venv = resettable_env.DictExtractWrapper(
        DummyVecEnv([lambda: gym.make(env_name)] * 4), "state"
    )
    return env, state_venv

def gen_expert_trajs(env, pi, state_venv):
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

    return expert_trajs

def irl_from_occupancy_measure(om, env, state_venv):
    ################ LEARN FROM OCCUPANCY MEASURE ##########################
    mce_irl_from_om, reward_net = train_mce_irl(om, env, state_venv)
    # mce_irl_from_om.plot_reward_map(reward_net, "om")

def irl_from_trajs(expert_trajs, env, state_venv):
    mce_irl_from_trajs, reward_net = train_mce_irl(expert_trajs, env, state_venv)
    # mce_irl_from_trajs.plot_reward_map(reward_net, "trajs")

def convert_statte_transition_to_features(transition, env):
    out = []

    for old, new in zip(env.to_coord(transition["old_state"]), env.to_coord(transition["new_state"])):
        out.append(old)
        out.append(new)

    return out

def convert_traj_to_features(traj, env):
    out_dim0 = []
    out_dim1 = []
    out_dim2 = []

    for state in traj:
        coords = env.to_coord(state)
        out_dim0.append(coords[0])
        out_dim1.append(coords[1])
        out_dim2.append(coords[2])

    out = out_dim0 + out_dim1 + out_dim2

    return out

    return out

def get_relevant_dim():

    env, env_only_cols, env_only_objs, state_venv, state_venv_only_cols, state_venv_only_objs = make_envs()

    _, _, pi_exp = mce_partition_fh(env, policy_temp=1)
    expert_trajs = gen_expert_trajs(env, pi_exp, state_venv)

    _, _, pi_rand = mce_partition_fh(env, policy_temp=100000000)
    rand_trajs = gen_expert_trajs(env, pi_rand, state_venv)

    X = []
    y = []

    USE_STATE_TRANSITIONS= False

    # append dxpert trajectories with y label 1
    for traj in expert_trajs:
        if USE_STATE_TRANSITIONS:
            for transition in traj.infos:
                X.append(convert_statte_transition_to_features(transition, env))
                y.append(list((1,)))
        else:
            X.append(convert_traj_to_features(traj.obs, env))
            y.append(list((1,)))

    # append random trajectories with y label 0
    for traj in rand_trajs:
        if USE_STATE_TRANSITIONS:
            for transition in traj.infos:
                X.append(convert_statte_transition_to_features(transition, env))
                y.append(list((0,)))
        else:
            X.append(convert_traj_to_features(traj.obs, env))
            y.append(list((0,)))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # both row and col coordinate
    clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1").fit(X_train, y_train)
    print(f"Test classification score both: {clf.score(X_test, y_test)}")

    # col only
    clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1").fit(X_train[:, :2], y_train)
    print(f"Test classification col only: {clf.score(X_test[:, :2], y_test)}")

    # row only
    clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1").fit(X_train[:, 2:4], y_train)
    print(f"Test classification row only: {clf.score(X_test[:, 2:4], y_test)}")

    # obj only
    clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l1").fit(X_train[:, 4:], y_train)
    print(f"Test classification row only: {clf.score(X_test[:, 4:], y_test)}")

def IRL_from_occupancy(om, env, state_venv):
    irl_from_occupancy_measure(om, env, state_venv)

def IRL_from_trajs(expert_trajs, env, state_venv):
    irl_from_trajs(expert_trajs, env, state_venv)


def get_envs_for_experiment(expert_env, learner_env):

    return *get_envs(expert_env), *get_envs(learner_env)

def modify_om_for_experiment(env_expert, env_learner, om):

    if not env_learner.reduce_rows and not env_learner.reduce_cols:
        return om
    elif env_learner.reduce_rows and not env_learner.reduce_cols:
        om = om.reshape(env_expert.height, env_expert.width, env_expert.n_levels)
        om = np.mean(om, axis=0)
        om = om.flatten()
        return om
    elif env_learner.reduce_rows and env_learner.reduce_cols:
        om = om.reshape(env_expert.height, env_expert.width, env_expert.n_levels)
        om = np.mean(om, axis=(0, 1))
        om = om.flatten()
        return om
    else:
        raise ValueError("Unknown Experiment Type")

def modify_expert_trajs_for_experiment(env_expert, env_learner, expert_trajs):

    if not env_learner.reduce_rows and not env_learner.reduce_cols:
        return expert_trajs
    elif env_learner.reduce_rows and not env_learner.reduce_cols:
        assert env_expert.n_objects == 0, "Not implemented for more levels yet"
        for traj in expert_trajs:
            traj.obs = [elem % env_expert.width for elem in traj.obs]
        return expert_trajs
    elif env_learner.reduce_rows and env_learner.reduce_cols:
        for traj in expert_trajs:
            traj.obs = [elem // (env_expert.width * env_expert.height) for elem in traj.obs]
        return expert_trajs
    else:
        raise ValueError("Unknown Experiment Type")

def print_run_info(expert_env, learner_env):
    print(f"####################### Expert: {expert_env} ######################")
    print(f"####################### Learner: {learner_env} ######################")

def run_IRL(expert_env_id, learner_env_id):

    print_run_info(expert_env_id, learner_env_id)

    env_expert, state_venv_expert, env_learner, state_venv_learner = get_envs_for_experiment(expert_env_id, learner_env_id)

    _, _, pi = mce_partition_fh(env_expert, policy_temp=7)

    _, om = mce_occupancy_measures(env_expert, pi=pi)
    ## TODO plot and save OM

    expert_trajs = gen_expert_trajs(env_expert, pi, state_venv_expert)

    om = modify_om_for_experiment(env_expert, env_learner, om)
    ## TODO save plot of modified OM

    IRL_from_occupancy(om, env_learner, state_venv_learner)
    ## TODO plot reward map

    expert_trajs = modify_expert_trajs_for_experiment(env_expert, env_learner, expert_trajs)

    IRL_from_trajs(expert_trajs, env_learner, state_venv_learner)
    ## TODO plot reward map
    ################## LEARN FROM OCCUPANCY MEASURE ##############################


    ## 33 return mean on full state space
    ## (high variance) return mean 25 on reduced state

    ################## LEARN FROM TRAJECTORIES ##############################


    # if EXPERT_TRAJS != "all":
    #     expert_trajs_passed = expert_trajs[0:EXPERT_TRAJS]
    # else:
    #     expert_trajs_passed = expert_trajs
    #
    # for traj in expert_trajs_passed:
    #     traj.obs = [elem // (env.width * env.height) for elem in traj.obs]

    ## 42 return mean on full state space
    ## (low variance) return mean 23 on reduced state

# def run_IRL_reduced_rows():
#
# def run_IRL_reduced_rows_cols():

def main():

   ## Experiment options:
   ## Expert_env: ["objects-0"]
   ## Learner_env: ["same", "reduced_rows", reduced_rows_reduced_cols"]

   run_IRL(expert_env_id="objects-0", learner_env_id="objects-0")
   run_IRL(expert_env_id="objects-0", learner_env_id="objects-0xred-rows")

   run_IRL(expert_env_id="objects-1", learner_env_id="objects-1")
   run_IRL(expert_env_id="objects-1", learner_env_id="objects-1xred-rowsxred-cols")

   run_IRL(expert_env_id="objects-3", learner_env_id="objects-3")
   run_IRL(expert_env_id="objects-3", learner_env_id="objects-3xred-rowsxred-cols")

def plot_om(om, id):
    plt.imshow(om)
    plt.savefig(f"om_{id}.png")

if __name__ == "__main__":
    # get_relevant_dim()
    main()

