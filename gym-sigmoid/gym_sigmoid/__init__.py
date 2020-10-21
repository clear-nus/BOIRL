import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SigmoidWorld-v0',
    entry_point='gym_sigmoid.envs:SigmoidWorldEnv',
    #timestep_limit=1000,
    reward_threshold=1.,
    nondeterministic = True,
)
