# Value iteration agent
# Model-based learning which requires mdp.
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import math
import numpy as np
import torch
import torch.nn.functional as F


def value_iteration(P_a, rewards, gamma, cudadevice,threshold = 1e-2, deterministic=True):
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
  N_STATES, N_ACTIONS,_ = np.shape(P_a)

  all_diffs = []
  vcopy = torch.zeros((N_STATES), dtype=torch.double)#.cuda(cudadevice)
  t_reward = torch.from_numpy(rewards)#.cuda(cudadevice)
  t_incidence = torch.from_numpy(P_a)#.cuda(cudadevice)
  diff = float("inf")
  while diff > threshold:
    # q = t_reward + discount * torch.mul(t_incidence, vcopy)
    q = torch.stack([t_reward] * N_ACTIONS, dim=1).double() + gamma * torch.reshape(
      torch.mm(torch.reshape(t_incidence, (-1, N_STATES)), torch.unsqueeze(vcopy, dim=1)), (N_STATES, N_ACTIONS))
    # q3 =  torch.mul(t_incidence, t_reward + discount*torch.stack([vcopy]*n_states,dim=0))
    v = torch.max(q,dim=1).values
    diff = torch.max(torch.abs(v - vcopy))
    all_diffs.append(diff)
    vcopy = v.clone()
  q = torch.stack([t_reward] * N_ACTIONS, dim=1).double() + gamma * torch.reshape(
    torch.mm(torch.reshape(t_incidence, (-1, N_STATES)), torch.unsqueeze(vcopy, dim=1)), (N_STATES, N_ACTIONS))
  # Uncomment this: q[7288:] = q[7288:] + t_incidence[7288:]
  # stoch_policy = F.softmax(q, dim=1)
  stoch_policy = F.softmax(q, dim=1)
  policy = None
  # stoch_policy = csr_matrix(stoch_policy.cpu().numpy())
  # q = csr_matrix(q.cpu().numpy())
  return policy, stoch_policy, v, q
