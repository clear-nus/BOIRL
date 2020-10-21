import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SwedenWorld-v0',
    entry_point='gym_sweden.envs:SwedenWorldEnv',
    #timestep_limit=1000,
    #reward_threshold=2.0,
    nondeterministic = True,
)
