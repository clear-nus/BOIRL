import numpy as np
import mdp.reward_value_iteration as vi

class GridWorld2D(object):
    def __init__(self,grid_size=6,goals = [7, 20, 31],discount=0.9,weights=[1.25, 5.0],horizon =15):#[0.,0.,100.,0.]): #goal 7, 28
        self.gtweights=np.array(weights)
        self.actions = ((1, 0), (0, 1),(-1, 0),(0, -1),(0, 0))
        self.n_actions = len(self.actions)
        self.N_ACTIONS = self.n_actions
        self.horizon = horizon
        self.grid_size = grid_size
        self.n_states = (grid_size**2)+1#last state is the sink state
        self.N_STATES = self.n_states
        self.discount = discount
        self.goals = goals
        self.weights = weights
        self.features = np.zeros((self.n_states))
        self.features[goals[0]] = 10. # high positive reward
        self.features[self.goals[1]] = 2. #close to zero reward
        self.features[self.goals[2]] = 1. #almost zero reward

        # Preconstruct the transition probability array.
        self.transition_probability = np.zeros((self.n_states,self.n_actions,self.n_states))
        for i in range(self.n_states-1):
            for j in range(self.n_actions):
                    self._better_transition_probability(i,j)
        #Automatic transition from any goal to sink state  regardless of action taken
        #Sink state only transitions to itself
        for g in self.goals:
            self.transition_probability[g,:,:] = 0
            self.transition_probability[g,:,self.grid_size**2] = 1
            self.transition_probability[self.grid_size**2,:,:] = 0
            self.transition_probability[self.grid_size**2,:,self.grid_size**2] = 1

        self.rewards = None
        self.set_reward(self.weights)

    def get_policy(self):
        policy, stoch_policy, v, q = vi.find_policy(self.n_states, self.n_actions, self.transition_probability,
                                                    self.rewards, self.discount, stochastic=False)
        return stoch_policy, q

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.
        i: State int.
        -> (x, y) int tuple.
        """
        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.
        p: (x, y) tuple.
        -> State int.
        """
        return p[0] + p[1]*self.grid_size

    def _better_transition_probability(self, i, j):
        x, y = self.int_to_point(i)
        kpoint = [x+self.actions[j][0],y+self.actions[j][1]]
        if (kpoint[0]<0 or kpoint[0]>=self.grid_size or kpoint[1]<0 or kpoint[1]>=self.grid_size):
            k = i
        else:
            k = self.point_to_int(kpoint)
            if k == self.grid_size**2 and i not in self.goals:
                k = i
        self.transition_probability[i,j,k] = 1


    def artificial_trajectories(self, trajectories, start_states):
        n_trajectories, l_trajectory, _ = np.shape(trajectories)
        art_trajectories = np.zeros((n_trajectories,l_trajectory,2),dtype=np.int)
        for t in range(n_trajectories):
            current_state = [start_states[t]]
            for l in range(l_trajectory):
                current_action = np.random.choice(np.arange(self.n_actions), size=1, p=np.ones(self.n_actions)/self.n_actions)
                art_trajectories[t, l, 0] = current_state[0]
                art_trajectories[t, l, 1] = current_action[0]
                next_state = np.random.choice(np.arange(self.n_states), size=1,
                                              p=self.transition_probability[current_state, current_action][0])
                current_state = next_state
        return art_trajectories

    def get_likelihood(self,trajectories):
        policy, loc_policy, v, q = vi.find_policy(self.n_states, self.n_actions, self.transition_probability,
                                                    self.rewards, self.discount, stochastic=False)
        a = np.sum(np.log(loc_policy[trajectories[:, :, 0], trajectories[:, :, 1]])) + np.sum(
            np.log(self.transition_probability[trajectories[:, :-1, 0], trajectories[:, :-1, 1], trajectories[:, 1:, 0]]))
        n_traj,l_traj,_ = np.shape(trajectories)

        likelihoods = -1. * a/ len(trajectories)
        return likelihoods, loc_policy

    def get_likelihood_from_policy(self,trajectories,loc_policy):
        a = np.sum(np.log(loc_policy[trajectories[:, :, 0], trajectories[:, :, 1]])) + np.sum(
            np.log(self.transition_probability[trajectories[:, :-1, 0], trajectories[:, :-1, 1], trajectories[:, 1:, 0]]))
        n_traj,l_traj,_ = np.shape(trajectories)

        likelihoods = -1. * a/ len(trajectories)
        return likelihoods

    def set_reward(self, w):
        self.weights = w
        rewards = 10./(1. + np.exp(-1.*w[0]*(self.features-w[1])))
        rewards[len(self.features)-1] = 0
        self.rewards = rewards

    def reset_reward(self):
        self.weights = np.array(self.gtweights)
        self.set_reward(self.weights)

    def evaluate_expsor(self, trajectories):
        n_trajectories, l_trajectory, _ = np.shape(trajectories)
        esor = 0.
        discounts = np.array([self.discount**t for t in range(l_trajectory)])
        discounts = np.repeat(np.expand_dims(discounts,axis=0),n_trajectories,0)
        just_states = trajectories[:,:,0]
        discounted_states = np.multiply(self.rewards[just_states],discounts)
        reward_trajectories = np.sum(discounted_states,axis=1)
        discounted_states = discounted_states.flatten()
        return np.sum(discounted_states)/n_trajectories,reward_trajectories

    def generate_trajectories (self,n_trajectories = 20, startpos=None,random_start = False):
        policy,stoch_policy, v, q = vi.find_policy(self.n_states,self.n_actions,self.transition_probability,self.rewards,self.discount,stochastic=False)
        start_states = np.zeros(n_trajectories,dtype=np.int)
        trajectories  = np.zeros((n_trajectories,self.horizon,2),dtype=np.int)
        sstate = np.arange(self.n_states-1)#(self.grid_size**2)
        for t in range(n_trajectories):
            current_state = startpos[t%len(startpos)] if (startpos is not None) else (np.random.randint(0,self.n_states,1) if random_start else np.array([sstate[t%(len(sstate))]]))
            if startpos is not None:
                current_state = np.array([int(current_state)])
            start_states[t] = current_state
            for l in range(self.horizon):
                current_action = np.random.choice(np.arange(self.n_actions),size=1,p=stoch_policy[current_state][0])#policy[current_state]#
                trajectories[t,l,0] = current_state[0]
                trajectories[t,l,1] = current_action[0]
                next_state = np.random.choice(np.arange(self.n_states),size=1,p=self.transition_probability[current_state,current_action][0])
                current_state = next_state
        
        return trajectories,start_states,stoch_policy

    def generate_trajectories_from_policy(self, stoch_policy, n_trajectories=20, startpos=None, random_start=False):
        start_states = np.zeros(n_trajectories, dtype=np.int)
        trajectories = np.zeros((n_trajectories, self.horizon, 2), dtype=np.int)
        sstate = np.arange(self.n_states - 1)  # (self.grid_size**2)
        for t in range(n_trajectories):
            current_state = startpos[t % len(startpos)] if (startpos is not None) else (
                np.random.randint(0, self.n_states, 1) if random_start else np.array([sstate[t % (len(sstate))]]))
            if startpos is not None:
                current_state = np.array([int(current_state)])
            start_states[t] = current_state
            for l in range(self.horizon):
                current_action = np.random.choice(np.arange(self.n_actions), size=1,
                                                  p=stoch_policy[current_state][0])  # policy[current_state]#
                trajectories[t, l, 0] = current_state[0]
                trajectories[t, l, 1] = current_action[0]
                next_state = np.random.choice(np.arange(self.n_states), size=1,
                                              p=self.transition_probability[current_state, current_action][0])
                current_state = next_state

        return trajectories, start_states, stoch_policy
