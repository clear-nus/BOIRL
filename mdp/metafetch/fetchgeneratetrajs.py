import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
import imageio
import time

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
from stable_baselines.common.vec_env import VecVideoRecorder

from mdp.metafetch.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model

# Fix for breaking change in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


def generate_trajectories(envid, algo, folder, exp_id, weights=None,indices= -1,sr=100):
    global KEY_ORDER

    env_id = envid
    algo = algo
    folder = folder
    seed = 0
    reward_log = ""
    no_render = True
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
    #hyperparams["env_wrapper"] = {"gym.wrappers.Monitor":{'directory':os.path.join(folder,str(exp_id))}}

    log_dir = reward_log if reward_log != '' else None

    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=not no_render,
                          hyperparams=hyperparams,weights = weights)

    """
    video_folder = 'logs/videos/'
    video_length = 100
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="random-agent-{}".format(env_id))
    """


    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    sim_init_state = env.envs[0].env.env.get_sim_initState()
    fullobs = None
    fulldesired = None
    fullachieved = None
    fullactions = None
    fulldone = []
    obs = env.reset()
    n_timesteps = 500
    verbose = 1

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not stochastic

    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    infosuc = 0.0

    """
    if not no_render:
        env.render('human')
        env.envs[0].unwrapped.viewer._record_video = True
    """
    if not no_render:
        env.render('human')
    #######################################################
    #if indices==0:
    #    time.sleep(10)
    #######################################################

    for iter in range(n_timesteps):
        if not no_render:
            pass
            #env.envs[0].env.env.viewer._run_speed = 0.5
            #env.envs[0].env.env.viewer.add_overlay(1, "Iteration: ", str(indices))
            #env.envs[0].env.env.viewer.add_overlay(1, "SuccessRate: ", "%3.2f%%"%sr)
            #env.envs[0].env.env.viewer.add_overlay(1, "Threshold: ", "%1.2f"%weights[0])
            #env.envs[0].env.env.viewer.add_overlay(1, "Penalty: ", "%1.2f"%weights[1])
        action, _ = model.predict(obs, deterministic=deterministic)
        if fullobs is None:
            fullobs = np.expand_dims(obs["observation"], axis=0)
            fulldesired = np.expand_dims(obs["desired_goal"], axis=0)
            fullachieved = np.expand_dims(obs["achieved_goal"], axis=0)
            fullactions = np.expand_dims(action, axis=0)
        else:
            fullobs = np.append(fullobs, np.expand_dims(obs["observation"], axis=0), axis=0)
            fulldesired = np.append(fulldesired, np.expand_dims(obs["desired_goal"], axis=0), axis=0)
            fullachieved = np.append(fullachieved, np.expand_dims(obs["achieved_goal"], axis=0), axis=0)
            fullactions = np.append(fullactions, np.expand_dims(action, axis=0), axis=0)
        fulldone.append(infosuc)

        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        #print("Is Done? %d"%(1 if done else 0))
        #print(infos, done)
        infosuc = infos[0]['is_success']
        if not no_render:
            img = env.render(mode='human')


        episode_reward += reward[0]
        ep_len += 1

        if n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                #print(done,not is_atari,verbose)
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                """
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                ep_len = 0
                """

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):

                if algo == 'her' and verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                ep_len = 0
                obs = env.reset()
                if algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0
        trajectories = {}
        trajectories["observation"] = np.array(fullobs).squeeze()
        trajectories["achieved_goal"] = np.array(fullachieved).squeeze()
        trajectories["desired_goal"] = np.array(fulldesired).squeeze()
        trajectories["indices"] = np.array(fulldone).squeeze()
        trajectories["actions"] = np.array(fullactions).squeeze()

    if verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not no_render:
        if n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()
    env.close()
    return trajectories,np.mean(successes),np.mean(episode_rewards)


