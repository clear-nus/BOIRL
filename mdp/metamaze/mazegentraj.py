import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import TRPO
import os
import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

envid = 'PointMazeLeft-v0'
savedir = "MazeTrainedPoliciesBD"
os.makedirs(savedir, exist_ok=True)


def gentraj(weights, ntrajectories, trial, algo, savefile=True):
    weights = np.append(weights,[0])
    algodir = os.path.join(savedir,algo)
    os.makedirs(algodir,exist_ok=True)
    np.save(os.path.join(algodir, "weights%d.npy") % trial, weights)
    allnll = []
    for t in range(1):
        model,trajs = train_maze(weights, ntrajectories, trial, savefile=savefile, attempt=t, algodir = algodir)
    return trajs


def train_maze(weights, ntrajectories, fid, algodir, savefile=True, attempt=1):
    #eargs = {"weights": weights}
    #env_temp = gym.make(envid, weights=weights)
    #env = make_vec_env(env_temp.__class__, env_kwargs=eargs, n_envs=8)
    #env = make_vec_env(envid, env_kwargs={"weights": weights}, n_envs=8)
    #env = DummyVecEnv([lambda: gym.make(envid)])
    env = gym.make(envid, weights=weights, randomstart=True)
    #model = PPO2(MlpPolicy, env, verbose=0, ent_coef=0, gamma=0.9)
    model = TRPO(MlpPolicy, env, verbose=0, gamma=0.9, n_cpu_tf_sess=8)
    model.learn(total_timesteps=300000)
    if savefile:
        model.save(os.path.join(algodir, "ppo2_pm_%d_%d" % (fid, attempt)))

    trajectories = []
    count = -1
    while len(trajectories) < ntrajectories:
        count += 1
        current_obs = None
        current_acts = None
        obs = env.reset()
        dones = False
        while not (dones):
            action = model.predict(obs)[0]  # model.predict(obs)[0]#env.action_space.sample()#
            current_obs = obs if current_obs is None else np.vstack((current_obs, np.array(obs)))
            current_acts = action if current_acts is None else np.vstack((current_acts, np.array(action)))
            obs, rewards, dones, info = env.step(action)
            dones = len(current_obs) >= 100
        current_traj = {"observations": current_obs, "actions": current_acts}
        trajectories.append(current_traj)
    env.close()
    return model, trajectories


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

def gen_traj_pol(algo,n,ntrajectories):
    env = gym.make(envid, randomstart=True)
    fid = os.path.join(savedir,algo+"_good1","ppo2_pm_%d_0.zip")%n
    model = TRPO.load(fid,env=env)#(MlpPolicy, env, verbose=0, gamma=0.9, n_cpu_tf_sess=8)
    trajectories = []
    count = -1
    while len(trajectories) < ntrajectories:
        count += 1
        current_obs = None
        current_acts = None
        obs = env.reset()
        dones = False
        while not (dones):
            action = model.predict(obs)[0]  # model.predict(obs)[0]#env.action_space.sample()#
            current_obs = obs if current_obs is None else np.vstack((current_obs, np.array(obs)))
            current_acts = action if current_acts is None else np.vstack((current_acts, np.array(action)))
            obs, rewards, dones, info = env.step(action)
            dones = len(current_obs) >= 100
        current_traj = {"observations": current_obs, "actions": current_acts}
        trajectories.append(current_traj)
    env.close()
    return model, trajectories
