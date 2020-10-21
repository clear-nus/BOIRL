# Value iteration agent
# Model-based learning which requires mdp.
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from scipy.special import softmax

def value_iteration(P_a, rewards, gamma, cudadevice, dirname, threshold=1e-3, deterministic=True):
    """
    static value iteration function. Perhaps the most useful function in this repo

    inputs:
      P_a         NxNxN_ACTIONS transition probabilities matrix -
                                P_a[s0, s1, a] is the transition prob of
                                landing at state s1 when taking action
                                a at state s0
      rewards     Nx1 matrix - rewards for all the states
      gamma       float - RL discount
      error       float - threshold for a stop
      deterministic   bool - to return deterministic policy or stochastic policy

    returns:
      values    Nx1 matrix - estimated values
      policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
    """
    N_ACTIONS = len(P_a)
    N_STATES = len(rewards)

    """
    Pa_torch = {}
    for a in range(N_ACTIONS):
        values = P_a[a].data
        indices = np.vstack((P_a[a].row, P_a[a].col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = P_a[a].shape
        Pa_torch[a] = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().cuda()
    """

    all_diffs = []
    vcopy = np.zeros((N_STATES))
    diff = float("inf")
    count = 0
    stime = timeit.default_timer()
    while diff > threshold:
        # Since this is a deterministic transition, I can do matrix multiplication instead of element wise multiplication
        q = np.zeros((N_STATES, N_ACTIONS))
        for a in range(N_ACTIONS):
            rows = P_a[a].row
            cols = P_a[a].col
            rows2 = np.argsort(rows)
            cols2 = cols[rows2]
            q[:,a] = rewards + gamma*vcopy[cols2]

        v = np.max(q, axis=1)
        diff = np.amax(np.abs(v - vcopy))
        all_diffs.append(diff)
        vcopy = np.array(v)
        count += 1
    #plt.plot(all_diffs)
    #plt.savefig(os.path.join(dirname, "value_iteration%f.png"%timeit.default_timer()), bbox_inches="tight")
    #plt.close("all")


    #q = torch.stack([t_reward] * N_ACTIONS, dim=1).double() + gamma * torch.reshape(
    #    torch.mm(torch.reshape(t_incidence, (-1, N_STATES)), torch.unsqueeze(vcopy, dim=1)), (N_STATES, N_ACTIONS))
    # Uncomment this: q[7288:] = q[7288:] + t_incidence[7288:]
    # stoch_policy = F.softmax(q, dim=1)
    stoch_policy = softmax(q, axis=1)
    policy = None
    # stoch_policy = csr_matrix(stoch_policy.cpu().numpy())
    # q = csr_matrix(q.cpu().numpy())
    return policy, stoch_policy, v, q
