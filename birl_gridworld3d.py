import numpy as np
from bayesian_irl.src.algos import learn, policy
from bayesian_irl.src.env import LoopEnv
from bayesian_irl.src.utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mdp.gridworld2d import GridWorld2D
import multiprocessing as mp
from boirlscenarios.irlobject import IRLObject
import boirlscenarios.constants as constants
from scipy import stats
from utils.plotme import plot_me
import timeit

def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=1, type=float, help='1/temperature of boltzmann distribution, '
                                                                     'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=55, type=int)
    parser.add_argument('--r_max', default=10, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--n_iter', default=2000, type=int)
    parser.add_argument('--burn_in', default=0, type=int)
    parser.add_argument('--dist', default='multiuniform', type=str,
                        choices=['uniform', 'gaussian', 'beta', 'gamma', 'multigauss', 'multiuniform'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq, ptrial, initpoints,savedir):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, ptrial, initpoints, savedir,**kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    weights = sample_random_rewards(env.n_states, step_size, r_max, ptrial, allrew=initpoints)
    env.set_reward(weights)
    # step 2
    # pi = learn.policy_iteration(env, gamma)
    pi, q = env.get_policy()
    # step 3
    stime = timeit.default_timer()
    timelist = [stime]
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
        current_time = timeit.default_timer()
        timelist.append(current_time)
        #if current_time-stime >= 600:
        #    break
    #np.save(os.path.join(savedir,"mytime%d.npy")%ptrial,np.array(timelist))


def is_not_optimal(q_values, pi):
    n_states, n_actions = q_values.shape
    for s in range(n_states):
        for a in range(n_actions):
            if q_values[s, np.argmax(pi[s])] < q_values[s, a]:
                return True
    return False


def compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
    ln_p_tilda = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma)
    ln_p = compute_posterior(demos, env, pi, prior, alpha, gamma)
    ratio = np.exp(ln_p_tilda - ln_p)
    return ratio


"""
def compute_posterior(demos, env, pi, prior, alpha, gamma):
    q = learn.compute_q_for_pi(env, pi, gamma)
    ln_p = np.sum([alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in demos]) + np.log(prior(env.weights))
    return ln_p
"""


def compute_posterior(demos, env, pi, prior, alpha, gamma):
    ln_p = np.sum([np.log(pi[s, a]) for s, a in demos]) + np.log(prior(env.weights))
    return ln_p


def mcmc_reward_step(weights, step_size, r_max):
    noweight = True
    while (noweight):
        new_weights = np.random.uniform((-2., -10., -4.), (2., 10., 4.), size=(1, 3)).squeeze()
        noweight = (np.linalg.norm(new_weights[0] - weights[0]) > step_size[0]) or (
            (np.linalg.norm(new_weights[1] - weights[1]) > step_size[1])) or (
                   (np.linalg.norm(new_weights[2] - weights[2]) > step_size[2]))

    """
    new_weights = copy.deepcopy(weights)
    index = np.random.randint(len(weights))
    step = np.random.choice([-step_size, step_size])
    new_weights[index] += step
    if np.all(new_weights == weights):
        new_weights[index] -= step
    #assert np.any(rewards != new_rewards), 'rewards do not change: {}, {}'.format(new_rewards, rewards)
    """
    return new_weights


def sample_random_rewards(n_states, step_size, r_max, ptrial, allrew):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    rewards = allrew[ptrial]
    # rewards = np.random.uniform(low=(-2,-10), high=(2,10), size=(1,2)).squeeze()
    # move these random rewards toward a gridpoint
    # add r_max to make mod to be always positive
    # add step_size for easier clipping
    """
    rewards = rewards + r_max + step_size
    for i, reward in enumerate(rewards):
        mod = reward % step_size
        rewards[i] = reward - mod
    # subtracts added values from rewards
    rewards = rewards - (r_max + step_size)
    """
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
        return prior()
    elif dist == "multiuniform":
        return prior()
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))


def main(args, t):
    np.random.seed(5)
    irlobj = IRLObject(kernel=constants.BIRL, env=constants.GRIDWORLD3D)
    demos = irlobj.fullTrajectories
    demos = demos.reshape((-1, 2))
    myinitpoints = np.load(os.path.join(irlobj.configurations.getTrajectoryDir(), "myinitpoints.npy"))
    saveprocdir = irlobj.configurations.getResultDir()
    # run birl
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(irlobj.env, demos, step_size=[0.05, 0.25, 0.1], n_iter=args.n_iter, r_max=args.r_max,
                                   prior=prior,
                                   alpha=args.alpha, gamma=irlobj.configurations.getDiscounts(), burn_in=args.burn_in,
                                   sample_freq=1, ptrial=t, initpoints=myinitpoints,savedir=saveprocdir)

    os.makedirs(saveprocdir, exist_ok=True)
    np.save(os.path.join(saveprocdir, "rewards%d.npy") % t, sampled_rewards)
    return sampled_rewards


def runmain(t):
    args = get_args()
    myweights = main(args, t)


# output = mp.Queue()
# Setup a list of processes that we want to run
total_trials = 10

for w in range(total_trials):
    runmain(w)

"""
processes = [mp.Process(target=runmain, args=(w, output)) for w in np.arange(2)]
# Run processes
for p in processes:
    p.start()
print("started")
# Exit the completed processes
for p in processes:
    print(p)
    p.join()
"""

# Plot the rewards
for t in tqdm(range(total_trials)):
    irlobj = IRLObject(kernel=constants.BIRL, env=constants.GRIDWORLD2D)
    current_reward = np.load(os.path.join(irlobj.configurations.getResultDir(), "rewards%d.npy") % t)
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
