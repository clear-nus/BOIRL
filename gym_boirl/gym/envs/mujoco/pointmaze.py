import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# from rllab.misc import logger

from gym.envs.mujoco.assets.mjc_models import point_mass_maze


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6, sparse_reward=False, no_reward=False, episode_length=100,
                 weights=None, initpos=None, randomstart=False):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length
        self.particle_initpos = initpos
        self.randomstart = randomstart
        """
        if randomstart:
            self.particle_initpos = np.random.uniform(low=[0., 0., 0.], high=[0.6, 0.6, 0.], size=(1, 3))[0].tolist()
        """
        # if weights is not None:
        #    print("Custom Weights")
        #    print(weights)
        self.weights = weights
        self.episode_length = 0

        model = point_mass_maze(direction=self.direction, length=self.length, initpos=self.particle_initpos)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def _step(self, a):
        if self.weights is None:
            vec_dist = self.get_body_com("particle") - self.get_body_com("target")
        else:
            vec_dist = self.get_body_com("particle") - self.weights
        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def step(self, action):
        return self._step(action)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        if self.randomstart:
            self.particle_initpos = np.random.uniform(low=[-0.05, -0.06, 0.], high=[0.55, 0.23, 0.], size=(1, 3))[0].tolist()
            self.sim.model.body_pos[1] = self.particle_initpos

        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle"),
            # self.get_body_com("target"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])


#O = PointMazeEnv(randomstart=True)
#O.reset()
#O.step(O.action_space.sample())
#O.reset()