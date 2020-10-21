import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import TRPO
import os
import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

envid = 'PointMazeLeft-v0'
savedir = "MazeTrainedPoliciesBD/"
os.makedirs(savedir, exist_ok=True)


def likelihood(weights, observations, actions, trial, algo, savefile=True):
    weights = np.append(weights,[0])
    algodir = os.path.join(savedir,algo)
    os.makedirs(algodir,exist_ok=True)
    np.save(os.path.join(algodir, "weights%d.npy") % trial, weights)
    allnll = []
    for t in range(1):
        model = train_maze(weights, trial, savefile=savefile, attempt=t, algodir = algodir)
        nll = single_likelihood(model, observations, actions)
        allnll.append(nll)
    allnll = np.array(allnll)
    np.save(os.path.join(algodir,"lik%d.npy")%trial,np.mean(allnll))
    return np.mean(allnll)


def train_maze(weights, fid, algodir, savefile=True, attempt=1):
    #eargs = {"weights": weights}
    #env_temp = gym.make(envid, weights=weights)
    #env = make_vec_env(env_temp.__class__, env_kwargs=eargs, n_envs=8)
    #env = make_vec_env(envid, env_kwargs={"weights": weights}, n_envs=8)
    #env = DummyVecEnv([lambda: gym.make(envid)])
    env = gym.make(envid, weights=weights, randomstart=True)
    #model = PPO2(MlpPolicy, env, verbose=0, ent_coef=0, gamma=0.9)
    model = TRPO(MlpPolicy, env, verbose=0, gamma=0.9, n_cpu_tf_sess=8)
    model.learn(total_timesteps=300000)
    model.save(os.path.join(algodir, "ppo2_pm_%d_%d" % (fid, attempt)))
    return model


def single_likelihood(model, obs, action):
    probs = []
    for ntraj in range(obs.shape[0]):
        current_prob = 0.
        for l in range(obs.shape[1]):
            current_prob += model.action_probability(observation=obs[ntraj, l], actions=action[ntraj, l],
                                                     logp=True).item()
        probs.append(-1 * current_prob)
    probs = np.array(probs)
    return np.mean(probs)
