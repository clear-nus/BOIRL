import gym
from gym import spaces
from gym import utils
import numpy as np
import logging
import numpy.random as rn
import math
from itertools import product
from scipy.sparse import load_npz
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse.coo import coo_matrix
import os

logger = logging.getLogger(__name__)


class SwedenWorldEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.basedir = "/home/user/basedir/" #fill in your path here.
        self.discount = 0.99
        self.sweden_dir = os.path.join(self.basedir,"mdp/metaborlange/newattempt2")
        self.destination = 7621  # Destination is in MATLAB indices. So remember to substract indices at right place
        self.horizon = 100
        
        self.incidence_smat_no_dummy = load_npz(os.path.join(self.sweden_dir, "incidence_no_dummy.npz"))
        self.incidence_smat_dummy = load_npz(os.path.join(self.sweden_dir, "incidence_dummy.npz"))
        self.incidence_smat = load_npz(os.path.join(self.sweden_dir, "incidence.npz"))
        self.travel_time_smat = load_npz(os.path.join(self.sweden_dir, "travel_time.npz"))
        self.turnangle_smat = load_npz(os.path.join(self.sweden_dir, "turn_angle.npz"))
        self.uturn_smat = load_npz(os.path.join(self.sweden_dir, "u_turn.npz"))
        self.lefturn_smat = load_npz(os.path.join(self.sweden_dir, "left_turn.npz"))
        self.observation_smat = load_npz(os.path.join(self.sweden_dir, "observation.npz"))

        self.gt_theta = np.array([-2., -1., -1., -20.])

        self.N_ACTIONS = 6  # 0-4 correspond to turn angles from -3.14 to 3.14. Action 5 corresponds to reaching destination
        self.N_ROADLINKS = self.travel_time_smat.shape[0]
        # self.goal_reward = 10.

        self.features = np.load(os.path.join(self.sweden_dir, "new_feat_data.npy"))
        self.transition_probabilities = self.getTP()
        self.state_debug = np.load(os.path.join(self.sweden_dir, "new_state_debug.npy"), allow_pickle=True).item()
        self.rewards = np.load(os.path.join(self.sweden_dir, "virtual_rewards.npy"))
        self.nodummy_states = np.array(np.load(os.path.join(self.sweden_dir, "nodummy_states.npy")))
        self.gt_theta = np.array([-2., -1., -1., -20.])
        self.N_STATES, self.N_FEATURES = np.shape(self.features)
        self.viewer = None
        self.server_process = None
        self.server_port = None

        self.state = None
        self.obs = None

        self.observation_space = spaces.Box(low=min(0., np.min(self.features)),
                                            high=max(1., np.max(self.features)),
                                            shape=(self.N_FEATURES,))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        # self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Box(low=0, high=self.N_ACTIONS,
                                       shape=(1,))
        self.reset()
        print("init over")

    def step(self, action):
        #action = (np.round(action)).astype(np.int)[0]
        #action2 = min(max(action, 0), self.N_ACTIONS)
        action2 = np.floor(action).astype(np.int)[0]
        action3 = min(max(action2, 0), self.N_ACTIONS-1)
        if not(action3<self.N_ACTIONS):
            print("This should not happen")
            print(action,action2,action3)
        obs, obsind, reward, done = self._take_action(action3)
        return obs, reward, done, {}

    def _take_action(self, action):
        currentTp = self.transition_probabilities[action]
        rowind = np.where(currentTp.row == self.state)[0]
        assert (len(rowind) > 0)
        next_state = currentTp.col[rowind].item()
        self.state = next_state
        self.obs = self.features[next_state]
        reward = self.rewards[next_state]
        if next_state == 20198:
            done=True
        else:
            done=False
        return self.obs, next_state, reward, done

    def reset(self):
        newState = np.random.choice(self.nodummy_states, 1).item()
        newObs = self.features[newState]
        self.state = newState
        self.obs = newObs
        return self.obs

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        raise NotImplementedError

    def getTP(self):
        transitions = np.load(os.path.join(self.sweden_dir, "new_transitions.npy"), allow_pickle=True).item()
        nstates, nfeatures = np.shape(self.features)
        transition_dynamics = {}
        for i in range(self.N_ACTIONS):
            tpsparse = coo_matrix((transitions[i][2, :], (transitions[i][0, :], transitions[i][1, :])),
                                  shape=(nstates, nstates))
            tpdense = tpsparse.toarray()
            assert (np.max(np.sum(tpdense, axis=1)) == 1. and np.min(np.sum(tpdense, axis=1)) == 1.)
            transition_dynamics[i] = coo_matrix((transitions[i][2, :], (transitions[i][0, :], transitions[i][1, :])),
                                                shape=(nstates, nstates))
        return transition_dynamics
