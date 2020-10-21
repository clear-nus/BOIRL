import gym
from gym import spaces
from gym import utils
import numpy as np
import logging
import numpy.random as rn
import math
from itertools import product


logger = logging.getLogger(__name__)


class SigmoidWorldEnv(gym.Env, utils.EzPickle):


    metadata = {'render.modes': ['human']}


    def __init__(self,grid_size=6,goals = [7, 20, 31],discount=0.9,weights=[1.25, 5.0, 0.],horizon =15):
        debug = True
        self.gtweights=np.array(weights)
        self.actions = ((1, 0), (0, 1),(-1, 0),(0, -1),(0, 0))
        self.n_actions = len(self.actions)
        self.horizon = horizon
        self.grid_size = grid_size
        self.n_states = (grid_size**2)+1#last state is the sink state
        self.discount = discount
        self.goals = goals
        self.weights = weights
        self.features = np.zeros((self.n_states))
        self.features[goals[0]] = 10. # high positive reward
        self.features[self.goals[1]] = 2. #close to zero reward
        self.features[self.goals[2]] = 1. #almost zero reward
        self.features[len(self.features)-1] = -50.
        #np.save("sigmoid_obs.npy",self.features)

        # Preconstruct the transition probability array.
        self.transition_probability = np.zeros((self.n_states,self.n_actions,self.n_states))
        for i in range(self.n_states-1):
            for j in range(self.n_actions):
                self._better_transition_probability(i,j)
        #Automatic transition from any goal to sink state  regardless of action taken
        #Sink state only transitions to itself
        for g in self.goals:
            self.transition_probability[g,:,:] = 0
            self.transition_probability[g,:,self.grid_size**2] = 1
            self.transition_probability[self.grid_size**2,:,:] = 0
            self.transition_probability[self.grid_size**2,:,self.grid_size**2] = 1
        self.rewards = None
        self.set_reward(self.weights)


        self.viewer = None
        self.server_process = None
        self.server_port = None

        self.observation_space = spaces.Box(low=np.min(self.features), high=np.max(self.features),
                                            shape=(1,))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        #self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Box(low=0, high=5,
                                       shape=(1,))
        self.reset()
        print("init over")

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.
        i: State int.
        -> (x, y) int tuple.
        """
        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.
        p: (x, y) tuple.
        -> State int.
        """
        return p[0] + p[1]*self.grid_size

    def _better_transition_probability(self, i, j):
        x, y = self.int_to_point(i)
        kpoint = [x+self.actions[j][0],y+self.actions[j][1]]
        if (kpoint[0]<0 or kpoint[0]>=self.grid_size or kpoint[1]<0 or kpoint[1]>=self.grid_size):
            k = i
        else:
            k = self.point_to_int(kpoint)
            if k == self.grid_size**2 and i not in self.goals:
                k = i
        self.transition_probability[i,j,k] = 1


    def set_reward(self, w):
        self.weights = w
        rewards = 10./(1. + np.exp(-1.*w[0]*(self.features-w[1])))
        #rewards[len(self.features)-1] = 0
        rewards += w[2]
        self.rewards = rewards

    def step(self, action):
        action = action.astype(np.int)[0]
        action = min(max(action,0),4)
        #print("#### Action: %d####" %action)
        obs,obsind = self._take_action(action)
        self.obs = obs
        self.obsind = obsind
        reward = self.rewards[obsind]
        episode_over = False
        #if obsind == self.n_states-1:
        #    episode_over = True

        return obs, reward, episode_over, {}

    def _take_action(self, action):
        next_state_prob = self.transition_probability[self.obsind,action]
        next_state_ind = np.random.choice(np.arange(self.n_states),1,p=next_state_prob)[0]
        next_state = self.features[next_state_ind]
        return next_state,next_state_ind


    def reset(self):
        stateind = np.random.randint(0,len(self.features),1)[0]
        obs = self.features[stateind]
        self.obs = obs
        self.obsind = stateind
        return obs

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        raise NotImplementedError
