import numpy as np
from algos import learn, policy
from env import LoopEnv
from utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mdp.borlangeworld import BorlangeWorld
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=1, type=float, help='1/temperature of boltzmann distribution, '
                                                                     'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=55, type=int)
    parser.add_argument('--r_max', default=10, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--n_iter', default=500, type=int)
    parser.add_argument('--burn_in', default=50, type=int)
    parser.add_argument('--dist', default='multiuniformborlange', type=str,
                        choices=['uniform', 'gaussian', 'beta', 'gamma', 'multigauss','multigaussBorlange','multiuniformborlange'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq,ptrial):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, ptrial, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    weights = sample_random_rewards(env.n_states, step_size, r_max, ptrial)
    env.set_reward(weights)
    # step 2
    # pi = learn.policy_iteration(env, gamma)
    pi, q = env.get_policy()
    # step 3
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        tilda_weights = mcmc_reward_step(env.weights, step_size, r_max)
        env_tilda.set_reward(tilda_weights)
        pi_tilda, q_pi_r_tilda = env_tilda.get_policy()
        # q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        if is_not_optimal(q_pi_r_tilda, pi):
            # pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            if np.random.random() < compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
                env, pi = env_tilda, pi_tilda
        else:
            if np.random.random() < compute_ratio(demos, env_tilda, pi, env, pi, prior, alpha, gamma):
                env = env_tilda
        yield env.weights


def is_not_optimal(q_values, pi):
    return np.any(
        q_values[np.arange(q_values.shape[0]).tolist(), np.argmax(pi, axis=1).tolist()] < np.argmax(q_values, axis=1))


def compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
    ln_p_tilda = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma)
    ln_p = compute_posterior(demos, env, pi, prior, alpha, gamma)
    ratio = np.exp(ln_p_tilda - ln_p)
    return ratio



def compute_posterior(demos, env, pi, prior, alpha, gamma):
    ln_p = np.sum([np.log(pi[s, a]) for s, a in demos]) + np.log(prior(env.weights))
    return ln_p


def mcmc_reward_step(weights, step_size, r_max):
    noweight = True
    while (noweight):
        new_weights = np.random.uniform((-2.5, -2.5), (0., 0.), size=(1, 2)).squeeze()
        noweight = (np.linalg.norm(new_weights[0] - weights[0]) > step_size[0]) or (
                np.linalg.norm(new_weights[1] - weights[1]) > step_size[1])
    return new_weights


def sample_random_rewards(n_states, step_size, r_max, ptrial):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    allrew = np.load("Data/vborlange/myinitpoints.npy")
    rewards = allrew[ptrial%allrew.shape[0],0:2]
    return rewards


def prepare_prior(dist, r_max):
    prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
    print(prior)
    if dist == 'uniform':
        return prior(xmax=r_max)
    elif dist == 'gaussian':
        return prior()
    elif dist in {'beta', 'gamma'}:
        return prior(loc=-r_max, scale=1 / (2 * r_max))
    elif dist == 'multigauss':
        return prior(dist)
    elif dist == "multigaussBorlange":
        return prior(dist)
    elif dist == "multiuniformborlange":
        return prior()
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))


def main(args,t):
    np.random.seed(5)
    env = BorlangeWorld(destination=7622, horizon=100,discount=0.99, loadres=True)
    demos = np.load(os.path.join("Data/vborlange/full_opt_trajectories.npy"))
    demos = demos.reshape((-1, 2))

    # run birl
    # prior = prepare_prior(args.dist, args.r_max)
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=[0.05, 0.05, 0.05], n_iter=args.n_iter, r_max=args.r_max,
                                   prior=prior,
                                   alpha=args.alpha, gamma=env.discount, burn_in=args.burn_in, sample_freq=1, ptrial=t)
    saveprocdir = "Results/Borlange"
    os.makedirs(saveprocdir, exist_ok=True)
    np.save(os.path.join(saveprocdir, "rewards%d.npy") % t, sampled_rewards)
    return sampled_rewards
    


def runmain(t,output):
    args = get_args()
    myweights = main(args,t)
    output.put((t,myweights))

output = mp.Queue()
# Setup a list of processes that we want to run
processes = [mp.Process(target=runmain, args=(w,output)) for w in np.arange(10)]
# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]
np.save("Borlange_parallel.npy",results)
