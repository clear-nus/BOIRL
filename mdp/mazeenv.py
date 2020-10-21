import numpy as np
import os
from mdp.metamaze.mazelikelihood import likelihood
from mdp.metamaze.mazegentraj import gentraj,gen_traj_pol
import gym


class MazeEnv:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def __init__(self):
        self.metadir = "mdp/metamaze"
        self.trajectories = np.load(os.path.join(self.metadir, "expert_trajectories.npy"), allow_pickle=True)
        self.observations = None
        self.actions = None
        for i in range(self.trajectories.shape[0]):
            current_obs = np.expand_dims(self.trajectories[i]["observations"], axis=0)
            current_acts = np.expand_dims(self.trajectories[i]["actions"], axis=0)
            self.observations = current_obs if self.observations is None else np.vstack(
                (self.observations, current_obs))
            self.actions = current_acts if self.actions is None else np.vstack((self.actions, current_acts))

        self.art_trajectories = None

        # self.art_trajectories = np.load(os.path.join(self.metadir, "random_trajectories.npy"), allow_pickle=True)
        self.art_observations = np.load(os.path.join(self.metadir, "art_traj_state.npy"))
        self.art_actions = np.load(os.path.join(self.metadir, "art_traj_action.npy"))
        """
        for i in range(self.art_trajectories.shape[0]):
            current_art_obs = np.expand_dims(self.art_trajectories[i]["observations"], axis=0)
            current_art_acts = np.expand_dims(self.art_trajectories[i]["actions"], axis=0)
            self.art_observations = current_art_obs if self.art_observations is None else np.vstack(
                (self.art_observations, current_art_obs))
            self.art_actions = current_art_acts if self.art_actions is None else np.vstack((self.art_actions, current_art_acts))
        """
        self.gtweights = np.array([0.3, 0.5])
        self.NTRAJ, self.LTRAJ = self.get_maxL()
        self.weights = np.array(self.gtweights)
        self.envid = 'PointMazeLeft-v0'
        self.algo = None
        self.trial = -1


    def set_reward(self, x):
        self.weights = x

    def get_likelihood(self, savefile=True):
        return likelihood(weights=self.weights, observations=self.observations, actions=self.actions, trial=self.trial,
                          savefile=savefile, algo=self.algo)
    def get_subs(self, n_traj, n_art):
        inds = np.random.permutation(self.NTRAJ)[0:n_traj]
        sub_trajs = self.observations[inds]
        sub_art_trajs = self.art_observations[:, inds]
        """
        for j in range(n_art):
            #inds = np.random.permutation(self.NTRAJ)[0:n_traj]
            current_trajs = np.expand_dims(self.art_observations[inds], axis=0)
            sub_art_trajs = current_trajs if sub_art_trajs is None else np.vstack((sub_art_trajs, current_trajs))
        """
        sub_spos = np.zeros(n_traj)
        return sub_trajs[:, :, 0:2], sub_spos, sub_art_trajs[:, :, :, 0:2]

    def get_maxL(self):
        N, L, _ = np.shape(self.observations)
        return N, L

    def generate_trajectories(self, n_trajectories=None, random_start=False, startpos=[]):
        trajs = gentraj(self.weights, n_trajectories, self.trial, self.algo, savefile=False)
        return trajs, None, None

    def generate_trajectories_from_policy(self, algo, n, n_trajectories=None):
        _,trajs = gen_traj_pol(algo,n,n_trajectories)
        return trajs, None, None

    def evaluate_expsor(self, trajs):
        allrew = []
        for t in trajs:
            obs = t["observations"]
            rew = -1 * np.linalg.norm(obs - np.append(self.weights,0),axis=1)
            disc = 0.9 ** (np.arange(len(obs)))
            rew2 = np.multiply(rew, disc)
            allrew.append(np.sum(rew2))
        return np.mean(allrew),None

    def get_random_states(self, nstates):
        env = gym.make(self.envid, randomstart=True)
        obs = env.reset()
        allstates = np.array(np.expand_dims(obs, axis=0))
        allactions = None
        for i in range(nstates):
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            action = np.expand_dims(action, axis=0)
            obs = np.expand_dims(obs, axis=0)
            allstates = np.vstack((allstates, obs))
            if allactions is None:
                allactions = action
            else:
                allactions = np.vstack((allactions, action))
        env.close()
        return allstates, allactions

    def get_reward(self, nstates):
        env = gym.make(self.envid, randomstart=True, weights=self.weights)
        obs = env.reset()
        allstates = np.array(np.expand_dims(obs, axis=0))
        # allactions = None
        allrewards = []
        for i in range(nstates):
            action = env.action_space.sample()
            obs, r, _, _ = env.step(action)
            action = np.expand_dims(action, axis=0)
            obs = np.expand_dims(obs, axis=0)
            allstates = np.vstack((allstates, obs))
            allrewards.append(r)
            """
            if allactions is None:
                allactions = action
            else:
                allactions = np.vstack((allactions, action))
            """
        env.close()
        return allstates[:-1], allrewards

    def get_reward_from_states(self, allStates):
        action = np.array([0., 0.])
        allr = []
        for o in allStates:
            env = gym.make(self.envid, randomstart=False, weights=self.weight, initpos=o)
            env.reset()
            otemp, rtemp, _, _ = env.step(action)
            allr.append(rtemp)
            env.close()

    def debug(self):
        weight = np.array([1., 1., 0.])
        obs = np.array([[1., 1., 0], [0., 0., 0.]])
        action = np.array([0., 0.])
        for o in obs:
            env = gym.make(self.envid, randomstart=False, weights=weight, initpos=o)
            env.reset()
            otemp, rtemp, _, _ = env.step(action)
        pass
