import os
import time
import difflib
import argparse
import importlib
import warnings
from pprint import pprint
from collections import OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import yaml
# Optional dependencies
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
try:
    import highway_env
except ImportError:
    highway_env = None

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None
from stable_baselines import HER, SAC, DDPG
from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.common.schedules import constfn
from mdp.metafetch.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class, find_saved_model
from mdp.metafetch.utils.hyperparams_opt import hyperparam_optimization
from mdp.metafetch.utils.noise import LinearNormalActionNoise
from mdp.metafetch.utils.utils import StoreDict
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
save_path = None
is_atari = None
env_id = None
algo_ = None
env_wrapper = None
normalize = None
normalize_kwargs = None
hyperparams = None
weights = None
seed = 0
verbose = 0

def create_env(n_envs, eval_env=False):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :param eval_env: (bool) Whether is it an environment used for evaluation or not
    :return: (Union[gym.Env, VecEnv])
    :return: (gym.Env)
    """
    global save_path
    global is_atari
    global env_id
    global algo_
    global env_wrapper
    global normalize
    global normalize_kwargs
    global hyperparams
    global weights
    global seed
    global verbose

    # Do not log eval env (issue with writing the same file)
    log_dir = None if eval_env else save_path

    if is_atari:
        if verbose > 0:
            print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif algo_ in ['dqn', 'ddpg']:
        if hyperparams.get('normalize', False):
            print("WARNING: normalization not supported yet for DDPG/DQN")
        env = gym.make(env_id)
        env.seed(seed)
        if env_wrapper is not None:
            env = env_wrapper(env)
    else:
        if n_envs == 1:
            env = DummyVecEnv(
                [make_env(env_id, 0, seed, wrapper_class=env_wrapper, log_dir=log_dir, weights=weights)])
        else:
            # env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([make_env(env_id, i, seed, log_dir=log_dir,
                                        wrapper_class=env_wrapper) for i in range(n_envs)])
        if normalize:
            if verbose > 0:
                if len(normalize_kwargs) > 0:
                    print("Normalization activated: {}".format(normalize_kwargs))
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **normalize_kwargs)
    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']
    return env

def train(current_envid, algo, current_weights):
    global save_path
    global is_atari
    global env_id
    global algo_
    global env_wrapper
    global normalize
    global normalize_kwargs
    global hyperparams
    global weights
    global seed
    global verbose


    log_folder = 'logs'
    seed = 0
    optimize = False
    verbose =0

    env_id = current_envid
    weights = current_weights
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(seed)


    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        seed += rank
        if rank != 0:
            verbose = 0

    tensorboard_log = None

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)
    print("Seed: {}".format(seed))

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(algo, env_id))

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = algo
    # HER is only a wrapper around an algo
    if algo == 'her':
        algo_ = saved_hyperparams['model_class']
        assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]
        if hyperparams['model_class'] is None:
            raise ValueError('{} requires MPI to be installed'.format(algo_))

    if verbose > 0:
        print(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if algo_ in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

    # Should we overwrite the number of timesteps?
    n_timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    # Convert to python object if needed
    if 'policy_kwargs' in hyperparams.keys() and isinstance(hyperparams['policy_kwargs'], str):
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])
        
    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = "{}/{}/".format(log_folder, algo)
    current_fid = get_latest_run_id(log_path, env_id) + 1
    save_path = os.path.join(log_path, "{}_{}".format(env_id, current_fid))
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)

    env = create_env(n_envs)
    # Stop env processes to free memory
    if optimize and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']

    if ALGOS[algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(algo))

    model = ALGOS[algo](env=env, tensorboard_log=tensorboard_log, verbose=verbose, **hyperparams)

    kwargs = {}


    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    print("Log path: {}".format(save_path))

    try:
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass


    # Only save worker of rank 0 when using mpi
    if rank == 0:
        print("Saving to {}".format(save_path))

        model.save("{}/{}".format(save_path, env_id))

        if normalize:
            # TODO: use unwrap_vec_normalize()
            # Unwrap
            if isinstance(env, VecFrameStack):
                env = env.venv
            # Important: save the running average, for testing the agent we need that normalization
            env.save(os.path.join(params_path, 'vecnormalize.pkl'))
            # Deprecated saving:
            # env.save_running_average(params_path)
    env.close()
    return current_fid
