"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.
Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np


def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.
    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-3):
    """
    Find the optimal value function.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    #horizon = 4
    v = np.zeros(n_states)
    #count = 0
    diff = float("inf")
    while diff > threshold:# and count < horizon:
        diff = 0
        vcopy = np.array(v)
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, reward[s] + discount * np.dot(tp, vcopy))
                #max_v = max(max_v, reward[s] + discount*np.dot(tp, f[s]*v))
                #max_v = max(max_v, reward[s] + discount * np.dot(tp, v))
            new_diff = abs(vcopy[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
        #count += 1
    return v

def optimal_value_sa(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-3):
    """
    Find the optimal value function.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        vcopy = np.array(v)
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, reward[s,a] + discount * np.dot(tp, vcopy))
                #max_v = max(max_v, reward[s] + discount*np.dot(tp, f[s]*v))
                #max_v = max(max_v, reward[s] + discount * np.dot(tp, v))
            new_diff = abs(vcopy[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
    return v




def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=False):
    """
    Find the optimal policy.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """
    #print "Editing Transition Probability"
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount,threshold)

    if stochastic:
        "WARNING: NEED TO MODIFY USING f(s)"
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(reward[s] + discount*transition_probabilities[s, a, k] *v[k]
                                     for k in range(n_states)))

    def _qvalue(s):
        qval = []
        for a in range(n_actions):
            qval.append(sum(reward[s] + discount*transition_probabilities[s, a, k] *v[k]
                                     for k in range(n_states)).item())
        #WHY DID I DO THE FOLLOWING TWO LINES!
        #qval = np.array(qval)
        #qval = np.exp(qval)/np.sum(np.exp(qval),axis=0)
        return qval

    def _stoch_policy(qs):
        max_qs = np.max(qs)
        qs = qs - max_qs
        policy = np.exp(qs)/np.sum(np.exp(qs))
        return policy



    policy = np.array([_policy(s) for s in range(n_states)])
    qvalues = np.array([_qvalue(s) for s in range(n_states)])
    stoch_policy = np.array([_stoch_policy(qvalues[s]) for s in range(n_states)])
    return policy,stoch_policy,v,qvalues
"""
if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
"""