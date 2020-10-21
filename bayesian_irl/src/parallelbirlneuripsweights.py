import numpy as np
from algos import learn, policy
from env import LoopEnv
from utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mdp.gridworld2d import GridWorld2D
import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=1, type=float, help='1/temperature of boltzmann distribution, '
                                                                      'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=55, type=int)
    parser.add_argument('--r_max', default=10, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--n_iter', default=10000, type=int)
    parser.add_argument('--burn_in', default=1000, type=int)
    parser.add_argument('--dist', default='multiuniform', type=str, choices=['uniform', 'gaussian', 'beta', 'gamma', 'multigauss', 'multiuniform'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq,ptrial):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, ptrial, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    weights = sample_random_rewards(env.n_states, step_size, r_max,ptrial)
    env.set_reward(weights)
    # step 2
    #pi = learn.policy_iteration(env, gamma)
    pi,q = env.get_policy()
    # step 3
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        tilda_weights = mcmc_reward_step(env.weights, step_size, r_max)
        env_tilda.set_reward(tilda_weights)
        pi_tilda, q_pi_r_tilda = env_tilda.get_policy()
        #q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        if is_not_optimal(q_pi_r_tilda, pi):
            #pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            if np.random.random() < compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
                env, pi = env_tilda, pi_tilda
        else:
            if np.random.random() < compute_ratio(demos, env_tilda, pi, env, pi, prior, alpha, gamma):
                env = env_tilda
        yield env.weights


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
    ln_p = np.sum([np.log(pi[s,a]) for s, a in demos]) + np.log(prior(env.weights))
    return ln_p



def mcmc_reward_step(weights, step_size, r_max):
    noweight = True
    while(noweight):
        new_weights = np.random.uniform((-2.,-10.),(2.,10.),size=(1,2)).squeeze()
        noweight = (np.linalg.norm(new_weights[0]-weights[0]) > step_size[0]) or ((np.linalg.norm(new_weights[1]-weights[1]) > step_size[1]))

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


def sample_random_rewards(n_states, step_size, r_max, ptrial):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    allrew = np.load("myinitpoints.npy")
    rewards = allrew[ptrial]
    #rewards = np.random.uniform(low=(-2,-10), high=(2,10), size=(1,2)).squeeze()
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
        return prior(loc=-r_max, scale=1/(2 * r_max))
    elif dist == 'multigauss':
        return prior()
    elif dist == "multiuniform":
        return prior()
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))

def main(args,t):
    np.random.seed(5)

    """
    # prepare environments
    if args.env_id == 0:
        env_args = dict(loop_states=[1, 3, 2])
    else:
        assert args.env_id == 1, 'Invalid env id is given'
        env_args = dict(loop_states=[0, 3, 2])
    env_args['rewards'] = [0, 0, 0.7, 0.7]
    env = LoopEnv(**env_args)

    # sample expert demonstrations
    expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    if args.policy == 'bol':
        expert_policy = policy.Boltzman(expert_q_values, args.alpha)
        print('pi \n', np.array([np.exp(args.alpha * expert_q_values[s])
                                 / np.sum(np.exp(args.alpha * expert_q_values[s]), axis=-1) for s in env.states]))
    else:
        expert_policy = policy.EpsilonGreedy(expert_q_values, epsilon=0.1)
    demos = np.array(list(sample_demos(env, expert_policy)))
    print('sub optimal actions {}/{}'.format(demos[:, 1].sum(), len(demos)))
    assert np.all(expert_q_values[:, 0] > expert_q_values[:, 1]), 'a0 must be optimal action for all the states'
    """
    env = GridWorld2D()
    #demos,_,_ = env.generate_trajectories(n_trajectories=200)
    demos = np.load("Data/gridworld2d/full_opt_trajectories.npy")
    demos = demos.reshape((-1,2))

    # run birl
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=[0.05,0.25], n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=env.discount, burn_in=args.burn_in, sample_freq=1, ptrial=t)
    saveprocdir = "Results/Gridworld"
    os.makedirs(saveprocdir,exist_ok=True)
    np.save(os.path.join(saveprocdir,"rewards%d.npy")%t, sampled_rewards)
    return sampled_rewards
    """
    np.save("Gridworld3d_rewards.npy",sampled_rewards)
    # plot rewards
    fig, ax = plt.subplots(1, env.n_states, sharey='all')
    for i, axes in enumerate(ax.flatten()):
        axes.hist(sampled_rewards[:, i], range=(-args.r_max, args.r_max))
    fig.suptitle('Loop Environment {}'.format(args.env_id), )
    path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-2], 'results',
                              'samples_env{}.png'.format(args.env_id))
    plt.savefig(path)

    est_rewards = np.mean(sampled_rewards, axis=0)
    print('True rewards: ', env_args['rewards'])
    print('Estimated rewards: ', est_rewards)

    # compute optimal q values for estimated rewards
    env.rewards = est_rewards
    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    for print_value in ('expert_q_values', 'learner_q_values'):
        print(print_value + '\n', locals()[print_value])
    print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))
    """

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
np.save("Gridworld_parallel.npy",results)