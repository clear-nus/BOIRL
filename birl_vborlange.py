import numpy as np
from bayesian_irl.src.algos import learn, policy
from bayesian_irl.src.env import LoopEnv
from bayesian_irl.src.utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mdp.borlangeworld import BorlangeWorld
import multiprocessing as mp
from boirlscenarios.irlobject import IRLObject
import boirlscenarios.constants as constants
from scipy import stats
from utils.plotme import plot_me


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
                        choices=['uniform', 'gaussian', 'beta', 'gamma', 'multigauss', 'multigaussBorlange',
                                 'multiuniformborlange'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq, ptrial,initpoints):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, ptrial, initpoints,**kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    weights = sample_random_rewards(env.n_states, step_size, r_max, ptrial,allrew=initpoints)
    env.set_reward(np.append(weights,[-1]))
    # step 2
    # pi = learn.policy_iteration(env, gamma)
    pi, q = env.get_policy()
    # step 3
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        tilda_weights = mcmc_reward_step(env.weights, step_size, r_max)
        env_tilda.set_reward(np.append(tilda_weights,[-1]))
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


def sample_random_rewards(n_states, step_size, r_max, ptrial, allrew):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    rewards = allrew[ptrial % allrew.shape[0], 0:2]
    return rewards


def prepare_prior(dist, r_max):
    prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
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


def main(args, t):
    np.random.seed(5)
    irlobj = IRLObject(kernel=constants.BIRL, env=constants.VIRTBORLANGE)
    demos = irlobj.fullTrajectories
    demos = demos.reshape((-1, 2))
    myinitpoints = np.load(os.path.join(irlobj.configurations.getTrajectoryDir(), "myinitpoints.npy"))

    # run birl
    # prior = prepare_prior(args.dist, args.r_max)
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(irlobj.env, demos, step_size=[0.05, 0.05, 0.05], n_iter=args.n_iter,
                                   r_max=args.r_max,
                                   prior=prior,
                                   alpha=args.alpha, gamma=irlobj.configurations.getDiscounts(), burn_in=args.burn_in,
                                   sample_freq=1, ptrial=t, initpoints=myinitpoints)
    saveprocdir = irlobj.configurations.getResultDir()
    os.makedirs(saveprocdir, exist_ok=True)
    np.save(os.path.join(saveprocdir, "rewards%d.npy") % t, sampled_rewards)
    return sampled_rewards


def runmain(t, output):
    args = get_args()
    myweights = main(args, t)
    output.put((t, myweights))

total_trials = 10
for w in range(total_trials):
    runmain(w)

"""
output = mp.Queue()
# Setup a list of processes that we want to run
processes = [mp.Process(target=runmain, args=(w, output)) for w in np.arange(10)]
# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]
"""

# Plot the rewards
for t in tqdm(range(total_trials)):
    irlobj = IRLObject(kernel=constants.BIRL, env=constants.GRIDWORLD2D)
    current_reward = np.load(os.path.join(irlobj.configurations.getResultDir(), "rewards%d.npy") % t)
    current_reward = current_reward.T
    xmin = irlobj.bounds[0]['domain'][0]
    xmax = irlobj.bounds[0]['domain'][1]
    ymin = irlobj.bounds[1]['domain'][0]
    ymax = irlobj.bounds[1]['domain'][1]

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    try:
        kernel = stats.gaussian_kde(current_reward)
        density = np.reshape(kernel(positions).T, X.shape)
        positions = positions.T
        plot_me(pos=positions, env=constants.GRIDWORLD2D, algo=constants.BIRL, val=density, notgreat=None, goodish=None,
                best=None, is_ours=False, savedir=irlobj.configurations.getResultDir(), fname="RewardDensity%d.png" % t,
                plt_xlabels=irlobj.configurations.plt_xlabels[constants.GRIDWORLD2D],
                plt_ylabels=irlobj.configurations.plt_ylabels[constants.GRIDWORLD2D],
                plt_xticks=irlobj.configurations.plt_xticks[constants.GRIDWORLD2D],
                plt_yticks=irlobj.configurations.plt_yticks[constants.GRIDWORLD2D],
                plt_xlims=irlobj.configurations.plt_xlims[constants.GRIDWORLD2D],
                plt_ylims=irlobj.configurations.plt_ylims[constants.GRIDWORLD2D],
                plt_gt=irlobj.configurations.plt_gt[constants.GRIDWORLD2D], trial=t, ismean=True)
        plt.close("all")
    except:
        print("Discarding the reward")
