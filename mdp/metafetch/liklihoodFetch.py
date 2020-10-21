import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
from scipy.stats import norm
from scipy.stats import multivariate_normal
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np

try:
    import highway_env
except ImportError:
    highway_env = None
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from mdp.metafetch.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model

# Fix for breaking change in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer


def getlikelihood(envid, algo, folder, exp_id, trajectories):
    global KEY_ORDER

    indices = trajectories["indices"]
    actions = trajectories["actions"]

    env_id = envid
    algo = algo
    folder = folder
    seed = 0
    reward_log = ""
    no_render = False
    deterministic = False
    stochastic = False
    norm_reward = False

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(exp_id))

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id)

    n_envs = 1

    set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    log_dir = reward_log if reward_log != '' else None

    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=not no_render,
                          hyperparams=hyperparams)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not stochastic

    inds = np.where(indices == 1.)[0]
    inds = np.append(inds, trajectories["observation"].shape[0]).astype(np.int)
    pol = 0.
    for i in np.arange(len(inds) - 1):
        traj_inds = np.arange(inds[i], inds[i + 1], 1)
        traj = np.concatenate([trajectories[key] for key in KEY_ORDER],axis=1)
        mu,std = model.policy_tf.proba_step(traj)
        for t in traj_inds:
            current_mu = mu[t]
            current_std = std[t]
            current_act = actions[t]
            mnorm = multivariate_normal(mean = current_mu,cov=current_std)
            pol += mnorm.logpdf(current_act)
    pol = -1*pol/(len(inds)-1)
    env.close()
    return pol

def getesor(envid, trajectories):
    global KEY_ORDER

    indices = trajectories["indices"]
    indices[0] = 1
    discount = 0.99
    env = gym.make(envid)
    env.reset()

    """
    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(exp_id))

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id)

    n_envs = 1

    set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    log_dir = reward_log if reward_log != '' else None

    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=not no_render,
                          hyperparams=hyperparams)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not stochastic

    """
    inds = np.where(indices == 1.)[0]
    inds = np.append(inds, trajectories["observation"].shape[0]).astype(np.int)
    rewards = []
    for i in np.arange(len(inds) - 1):
        traj_reward = 0.
        traj_inds = np.arange(inds[i], inds[i + 1], 1)
        for tind_in, t in enumerate(traj_inds):
            ach_goal = trajectories["achieved_goal"][t]
            des_goal = trajectories["desired_goal"][t]
            traj_reward += (discount**tind_in)*env.compute_reward(ach_goal,des_goal,None)
        rewards.append(traj_reward)
    rewards = np.array(rewards)
    sor = np.sum(rewards)/(len(inds)-1)
    env.close()
    return sor



