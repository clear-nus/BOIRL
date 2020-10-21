from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
import roboschool
#import gym_ow
import gym_sweden
import gym

class PointEnv(Env):
    def __init__(self):
        self.env = gym.make('SwedenWorld-v0')
        super().__init__()

    @property
    def observation_space(self):
        return Box(low = self.env.observation_space.low[0],high=self.env.observation_space.high[0],shape=self.observation_space.shape)

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()




