import numpy as np
from mdp.metaborlange.routedata import Routedata
from mdp.metaborlange import utils
from mdp.metaborlange import config
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from mdp.coo_value_iteration_parallel import value_iteration
import timeit
import os
import pickle
from tqdm import tqdm

class BorlangeWorld():
    def __init__(self,destination,discount,horizon=100,loadres=False,saveres=False):
        self.horizon = horizon
        self.discount = discount
        self.borlange_dir = "mdp/metaborlange/newattempt2"
        if not os.path.exists(self.borlange_dir):
            os.makedirs(self.borlange_dir)

        self.destination = destination -1#Destination is in MATLAB indices. So remember to substract indices at right place
        if loadres:
            self.incidence_smat_no_dummy = load_npz(os.path.join(self.borlange_dir,"incidence_no_dummy.npz"))
            self.incidence_smat_dummy = load_npz(os.path.join(self.borlange_dir,"incidence_dummy.npz"))
            self.incidence_smat = load_npz(os.path.join(self.borlange_dir,"incidence.npz"))
            self.travel_time_smat = load_npz(os.path.join(self.borlange_dir,"travel_time.npz"))
            self.turnangle_smat = load_npz(os.path.join(self.borlange_dir,"turn_angle.npz"))
            self.uturn_smat = load_npz(os.path.join(self.borlange_dir,"u_turn.npz"))
            self.lefturn_smat = load_npz(os.path.join(self.borlange_dir,"left_turn.npz"))
            self.observation_smat = load_npz(os.path.join(self.borlange_dir,"observation.npz"))
        else:
            self.observation_smat = None
            self.rd = Routedata()
            other_incidence = np.loadtxt("mdp/metaborlange/data/linkIncidence_full.txt")

            # load from matlab text files
            self.incidence_smat_no_dummy, self.incidence_smat_dummy \
                = self.rd.load_incidence_data(config.incidence_filename)
            self.incidence_smat = utils.convert_to_sparse_mat(other_incidence)

            self.travel_time_smat = self.rd.load_travel_time_data(
                config.travel_time_filename)
            self.turnangle_smat = self.rd.load_turn_angle_data(config.turn_angle_filename)
            self.uturn_smat = self.rd.get_uturn_feature_from_turn_angle()
            self.lefturn_smat = self.rd.get_lefturn_feature_from_turn_angle()
            if saveres:
                save_npz(os.path.join(self.borlange_dir,"incidence_no_dummy.npz"),self.incidence_smat_no_dummy)
                save_npz(os.path.join(self.borlange_dir,"incidence_dummy.npz"), self.incidence_smat_dummy)
                save_npz(os.path.join(self.borlange_dir,"incidence.npz"), self.incidence_smat)
                save_npz(os.path.join(self.borlange_dir,"travel_time.npz"), self.travel_time_smat)
                save_npz(os.path.join(self.borlange_dir,"turn_angle.npz"), self.turnangle_smat)
                save_npz(os.path.join(self.borlange_dir,"u_turn.npz"), self.uturn_smat)
                save_npz(os.path.join(self.borlange_dir,"left_turn.npz"), self.lefturn_smat)

        self.N_ACTIONS = 6 #0-4 correspond to turn angles from -3.14 to 3.14. Action 5 corresponds to reaching destination
        self.N_ROADLINKS = self.travel_time_smat.shape[0]
        #self.goal_reward = 10.
        self.features = None
        self.transition_probability = None
        self.state_debug = None
        self.rewards = None
        self.nodummy_states = None
        self.weights = np.array([-2., -1., -1.])
        #self.gtweights = np.array([-2., -1., -1.])
        #self.gtweights = np.array([-2., -1., -1., -20.])
        self.__set_featdata(loadres)
        self.set_reward(self.weights)
        self.N_STATES,self.N_FEATURES = np.shape(self.features)
        self.n_states = self.N_STATES
        self.n_actions = self.N_ACTIONS

    def gather_real_trajectories(self):
        if self.observation_smat is None:
            self.observation_smat = self.rd.load_observation_full_data(config.observation_filename)
            save_npz("observation.npz", self.observation_smat)


        valids = self.observation_smat[:, 0] == (self.destination+1)
        valid_trajs = []
        for i in range(np.shape(valids)[0]):
            cols = valids.indices[valids.indptr[i]:valids.indptr[i+1]]
            if len(cols) >0:
                valid_trajs.append(i)
        trajectories = self.observation_smat[valid_trajs,1:]
        trajectories = csr_matrix.toarray(trajectories)
        trajectories = (trajectories-1).astype(int)

        newFormatTrajectories = self.change_format(trajectories)
        return newFormatTrajectories, newFormatTrajectories[:,0,0]

    def generate_trajectories(self, n_trajectories=100, startpos = None, random_start=False):
        l_trajectories = self.horizon
        policy, stoch_policy, v, qvalues = value_iteration(self.transition_probability, self.rewards, self.discount,
                                                           None, self.borlange_dir)
        start_states = np.zeros(n_trajectories, dtype=np.int)
        trajectories = np.zeros((n_trajectories, l_trajectories, 2), dtype=np.int)
        if startpos is None:
            if random_start:
                sstate = np.random.randint(low = 0, high= np.max(self.nodummy_states), size=n_trajectories)
            else:
                sstate = self.nodummy_states#np.load("starting_points.npy")  # (self.grid_size**2)
        else:
            sstate = startpos
        #for t in tqdm(range(n_trajectories),desc="Expert Traj"):
        for t in tqdm(range(n_trajectories)):
            current_state = np.array([sstate[t % (len(sstate))]])
            start_states[t] = current_state
            for l in range(l_trajectories):
                current_action = np.random.choice(np.arange(self.N_ACTIONS), size=1,
                                                  p=stoch_policy[current_state][0])  # policy[current_state]#
                trajectories[t, l, 0] = current_state[0]
                trajectories[t, l, 1] = current_action[0]
                current_tp = self.transition_probability[current_action.item()]
                ind = np.where(current_tp.row == current_state.item())
                next_state = current_tp.col[ind]
                current_state = np.array(next_state)
        return trajectories,start_states, stoch_policy

    def generate_trajectories_from_policy(self, stoch_policy, n_trajectories=100, startpos = None, random_start=False):
        l_trajectories = self.horizon
        #policy, stoch_policy, v, qvalues = value_iteration(self.transition_probability, self.rewards, self.discount,
        #                                                   None, self.borlange_dir)
        start_states = np.zeros(n_trajectories, dtype=np.int)
        trajectories = np.zeros((n_trajectories, l_trajectories, 2), dtype=np.int)
        if startpos is None:
            if random_start:
                sstate = np.random.randint(low = 0, high= np.max(self.nodummy_states), size=n_trajectories)
            else:
                sstate = self.nodummy_states#np.load("starting_points.npy")  # (self.grid_size**2)
        else:
            sstate = startpos
        #for t in tqdm(range(n_trajectories),desc="Expert Traj"):
        for t in range(n_trajectories):
            current_state = np.array([sstate[t % (len(sstate))]])
            start_states[t] = current_state
            for l in range(l_trajectories):
                current_action = np.random.choice(np.arange(self.N_ACTIONS), size=1,
                                                  p=stoch_policy[current_state][0])  # policy[current_state]#
                trajectories[t, l, 0] = current_state[0]
                trajectories[t, l, 1] = current_action[0]
                current_tp = self.transition_probability[current_action.item()]
                ind = np.where(current_tp.row == current_state.item())
                next_state = current_tp.col[ind]
                current_state = np.array(next_state)
        return trajectories,start_states, stoch_policy

    def artificial_trajectories(self, trajectories, start_states):
        n_trajectories, l_trajectory, _ = np.shape(trajectories)
        art_trajectories = np.zeros((n_trajectories, l_trajectory, 2), dtype=np.int)
        for t in range(n_trajectories):
            current_state = np.array([start_states[t]])
            for l in range(l_trajectory):
                if l > -1:
                    current_action = np.random.choice(np.arange(self.N_ACTIONS), size=1,
                                                      p=np.ones(self.N_ACTIONS) / self.N_ACTIONS)
                else:
                    assert(current_state == trajectories[t,l,0])
                    current_action = trajectories[t,l,1:2]
                art_trajectories[t, l, 0] = current_state[0]
                art_trajectories[t, l, 1] = current_action[0]
                current_tp = self.transition_probability[current_action.item()]
                ind = np.where(current_tp.row == current_state.item())
                next_state = current_tp.col[ind]
                current_state = np.array(next_state)
        return art_trajectories

    def set_reward(self,w):
        self.weights = np.array(w)
        w = np.append(w,[-20])
        self.rewards = np.dot(self.features,w)

    def get_likelihood(self,trajectories,trial=0):
        policy, loc_policy, v, q = value_iteration(self.transition_probability, self.rewards, self.discount,
                                                   None, self.borlange_dir)
        a = np.sum(np.log(loc_policy[trajectories[:, :, 0], trajectories[:, :, 1]]))
        a = np.nan_to_num(a)
        n_traj, l_traj, _ = np.shape(trajectories)
        likelihoods = -1. * a / len(trajectories)
        return likelihoods, loc_policy

    def get_likelihood_from_policy(self,trajectories,loc_policy):
        a = np.sum(np.log(loc_policy[trajectories[:, :, 0], trajectories[:, :, 1]]))
        a = np.nan_to_num(a)
        n_traj, l_traj, _ = np.shape(trajectories)
        likelihoods = -1. * a / len(trajectories)
        return likelihoods

    def get_likelihood2(self,trajectories):
        policy, loc_policy, v, q = value_iteration(self.transition_probability, self.rewards, self.discount,
                                                   None, self.borlange_dir)
        n_trajectories, l_trajectory, _ = np.shape(trajectories)
        a = 0.
        for n in range(n_trajectories):
            indme = np.where(trajectories[n,:,0]==20198)
            if len(indme) == 0:
                indme = indme[0][0]
            else:
                indme = l_trajectory
            a += np.sum(np.log(loc_policy[trajectories[n, 0:indme, 0], trajectories[n, 0:indme, 1]]))
        a = np.nan_to_num(a)
        likelihoods = -1. * a / len(trajectories)
        return likelihoods

    def dummy_get_likelihood_from_policy(self,stoch_policy,trajectories):
        fullsize = self.incidence_smat.shape[0]
        demos = np.where(trajectories == 0., fullsize * np.ones(np.shape(trajectories)), trajectories - 1)
        """
        if np.all(self.incidence_smat_no_dummy.indices == self.travel_time_smat.indices) and np.all(
                self.incidence_smat_no_dummy.indptr == self.travel_time_smat.indptr):
            if np.all(self.incidence_smat_no_dummy.indices == self.uturn_smat.indices) and np.all(
                    self.incidence_smat_no_dummy.indptr == self.uturn_smat.indptr):
                if np.all(self.incidence_smat_no_dummy.indices == self.lefturn_smat.indices) and np.all(
                        self.incidence_smat_no_dummy.indptr == self.lefturn_smat.indptr):
                    feats = np.append(np.expand_dims(self.travel_time_smat.data,axis=0),np.expand_dims(self.uturn_smat.data,axis=0),axis=0)
                    feats = np.append(feats,np.expand_dims(self.lefturn_smat.data,axis=0),axis=0)
                else:
                    raise(NotImplementedError)
            else:
                raise (NotImplementedError)
        else:
            raise (NotImplementedError)


        Rx_full = np.dot(self.reward,feats)

        demos = np.where(trajectories == 0., fullsize * np.ones(np.shape(trajectories)), trajectories-1)

        Rx_meat = csr_matrix((Rx_full,self.incidence_smat_no_dummy.indices,self.incidence_smat_no_dummy.indptr),shape=self.incidence_smat_no_dummy.shape)
        n_meaty = Rx_meat.shape[0]

        #Calculate Reward for given weight
        Rx = Rx_meat
        full_size_extended = fullsize + 1
        Rx = vstack([Rx, csr_matrix((full_size_extended - n_meaty, n_meaty))], format="csr")
        Rx = hstack([Rx, csr_matrix((full_size_extended, int(self.destination) - n_meaty))], format="csr")
        Rx = hstack([Rx, csr_matrix(self.goal_reward * np.ones((full_size_extended, 1)))], format="csr")
        Rx = hstack([Rx, csr_matrix((full_size_extended, full_size_extended - Rx.shape[1]))], format="csr")

        policy, stoch_policy, v, qvalues = find_policy(n_states=self.n_states, n_actions=self.n_actions,
                                                       transition_probabilities=self.incidence_smat, reward=Rx,
                                                       discount=self.discount)
        """
        stoch_policy = hstack((stoch_policy,np.ones((stoch_policy.shape[0],1))),format="csr")
        bottom = np.zeros((1,stoch_policy.shape[1]))
        bottom[0,-1] = 1.
        stoch_policy = vstack((stoch_policy,csr_matrix(bottom)),format="csr")


        #stime2 = timeit.default_timer()
        #Expert Trajectory
        demo_states = demos[:,:-1].astype(int)
        demo_actions = demos[:,1:].astype(int)
        a = np.sum(np.log(csr_matrix.toarray(stoch_policy[demo_states,demo_actions])))
        n_traj, l_traj = np.shape(trajectories)
        likelihoods = -1. * a / len(trajectories)
        return likelihoods

    def save_obj(self, obj, name):
        with open(os.path.join(self.borlange_dir,'obj/' + name + '.pkl'), 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self,name):
        with open(os.path.join(self.borlange_dir,'obj/' + name + '.pkl'), 'rb') as f:
            return pickle.load(f)

    def __set_featdata(self,loadres=False):
        #Features: <Time, Left-Turn, CrossLinks, U-Turn>
        feat_data = None
        transitions = {0:None,
                       1:None,
                       2:None,
                       3:None,
                       4:None,
                       5:None}

        inc_dense = self.incidence_smat.toarray()
        row,col = np.shape(inc_dense)



        if not loadres:
            count = 0
            state_debug = {}
            nodummy_states = []
            #Get the feature data
            print("Getting the state features")
            for c in tqdm(np.arange(col)):
                if c < self.N_ROADLINKS:
                    inds = np.where(inc_dense[:,c]==1)[0]
                    inds_nodummy = inds[np.where(inds<self.N_ROADLINKS)[0]]
                    inds_dummy = inds[np.where(inds>=self.N_ROADLINKS)[0]]
                    assert(len(inds_dummy)==0)

                    #Get current turn angles of all the road links
                    #current_turnangles = self.turnangle_smat[inds_nodummy,c].toarray().squeeze()
                    #sorted_incoming  = np.argsort(current_turnangles)
                    for incoming_state in inds_nodummy:
                        state_uturn = self.uturn_smat[incoming_state,c]
                        state_leftturn = self.lefturn_smat[incoming_state,c]
                        state_time = self.travel_time_smat[incoming_state,c]
                        state_crosslinks = 1#len(self.incidence_smat[c,:].data)
                        if feat_data is None:
                            feat_data= np.array([[state_time,state_leftturn,state_crosslinks,state_uturn]])
                        else:
                            feat_data = np.append(feat_data,np.array([[state_time,state_leftturn,state_crosslinks,state_uturn]]),axis=0)

                        if c in state_debug:
                            tostore = np.array(state_debug[c])
                            tostore = np.append(tostore, np.array([[count, incoming_state]]), axis=0)
                            state_debug[c] = tostore
                        else:
                            tostore = np.array([[count, incoming_state]])
                            state_debug[c] = tostore

                        nodummy_states.append(count)
                        count += 1
                else:
                    if c == self.destination:
                        inds = np.where(inc_dense[:, c] == 1)[0]
                        placehold = 0.
                        for incoming_state in inds:
                            state_uturn = placehold
                            state_leftturn = placehold
                            state_time = placehold
                            state_crosslinks = placehold
                            if feat_data is None:
                                feat_data = np.array([[state_time, state_leftturn, state_crosslinks, state_uturn]])
                            else:
                                feat_data = np.append(feat_data,
                                                      np.array([[state_time, state_leftturn, state_crosslinks, state_uturn]]),
                                                      axis=0)
                            if c in state_debug:
                                tostore = np.array(state_debug[c])
                                tostore = np.append(tostore, np.array([[count, incoming_state]]),axis=0)
                                state_debug[c] = tostore
                            else:
                                tostore = np.array([[count, incoming_state]])
                                state_debug[c] = tostore
                            count += 1
                if c%100==0:
                    np.save(os.path.join(self.borlange_dir,"new_feat_data.npy"),feat_data)
                    np.save(os.path.join(self.borlange_dir,"new_state_debug.npy"),state_debug)
                    np.save(os.path.join(self.borlange_dir,"nodummy_states.npy"),np.array(nodummy_states))
            np.save(os.path.join(self.borlange_dir, "new_feat_data.npy"), feat_data)
            np.save(os.path.join(self.borlange_dir, "new_state_debug.npy"), state_debug)
            np.save(os.path.join(self.borlange_dir, "nodummy_states.npy"), np.array(nodummy_states))
        else:
            feat_data = np.load(os.path.join(self.borlange_dir, "new_feat_data.npy"))
            state_debug = np.load(os.path.join(self.borlange_dir, "new_state_debug.npy"),allow_pickle=True).item()
            nodummy_states = np.load(os.path.join(self.borlange_dir, "nodummy_states.npy"))



        #feat_data = np.load(os.path.join(self.borlange_dir,"new_feat_data.npy"))
        #state_debug = np.load(os.path.join(self.borlange_dir,"new_state_debug.npy"),allow_pickle=True).item()

        #Get transition probability

        if not loadres:
            print("Getting the transition dynamics")
            for r in tqdm(state_debug):
                corresponding_states = state_debug[r][:,0]
                badinds = np.where(inc_dense[r,:] == 1)[0]
                inds = []
                for i in badinds:
                    if i in state_debug:
                        inds.append(i)
                inds = np.array(inds)
                if r < self.N_ROADLINKS:
                    inds_nodummy = inds[np.where(inds < self.N_ROADLINKS)[0]]
                    inds_dummy = inds[np.where(inds >= self.N_ROADLINKS)[0]]
                    assert(len(inds_dummy)<=1)
                    current_turnangles = self.turnangle_smat[r, inds_nodummy].toarray().squeeze()
                    sorted_incoming  = np.argsort(current_turnangles)

                    for cs in corresponding_states:
                        for a in np.arange(self.N_ACTIONS - 1):
                            #print("Current R: %d, Current CS: %d, Action %d" %(r,cs,a))
                            if a < len(sorted_incoming):
                                next_state =  inds_nodummy[sorted_incoming[a]]
                                corresponding_next_states = state_debug[next_state]
                                temp_next_statenumber = np.where(corresponding_next_states[:,1]==r)[0]
                                assert(len(temp_next_statenumber)>0)
                                exact_next_state_number = corresponding_next_states[temp_next_statenumber,0].item()
                                tostoretp = transitions[a]
                                if tostoretp is None:
                                    tostoretp = np.array([[cs],[exact_next_state_number], [1.]])
                                else:
                                    tostoretp = np.array(tostoretp)
                                    tostoretp = np.append(tostoretp,np.array([[cs],[exact_next_state_number], [1.]]),axis=1)
                                transitions[a] = tostoretp
                            else:
                                tostoretp = transitions[a]
                                if tostoretp is None:
                                    tostoretp = np.array([[cs],[cs], [1.]])
                                else:
                                    tostoretp = np.array(tostoretp)
                                    tostoretp = np.append(tostoretp,np.array([[cs],[cs], [1.]]),axis=1)
                                transitions[a] = tostoretp
                        if len(inds_dummy)==0:
                            next_state = r
                            corresponding_next_states = state_debug[next_state]
                            temp_next_statenumber = np.where(corresponding_next_states[:, 1] == r)[0]
                            if len(temp_next_statenumber) > 0:
                                exact_next_state_number = corresponding_next_states[temp_next_statenumber, 0].item()
                            else:
                                exact_next_state_number = cs
                            tostoretp = transitions[5]
                            if tostoretp is None:
                                tostoretp = np.array([[cs], [exact_next_state_number], [1.]])
                            else:
                                tostoretp = np.array(tostoretp)
                                tostoretp = np.append(tostoretp, np.array([[cs], [exact_next_state_number], [1.]]), axis=1)
                            transitions[5] = tostoretp
                        else:
                            next_state = inds_dummy[0]
                            corresponding_next_states = state_debug[next_state]
                            exact_next_state_number = corresponding_next_states[
                                np.where(corresponding_next_states[:, 1] == r), 0].squeeze().item()
                            tostoretp = transitions[5]
                            if tostoretp is None:
                                tostoretp = np.array([[cs], [exact_next_state_number], [1.]])
                            else:
                                tostoretp = np.array(tostoretp)
                                tostoretp = np.append(tostoretp, np.array([[cs], [exact_next_state_number], [1.]]), axis=1)
                            transitions[5] = tostoretp
                else:
                    assert(len(inds)==1)
                    for cs in corresponding_states:
                        for a in np.arange(self.N_ACTIONS):
                            next_state = inds[0]
                            corresponding_next_states = state_debug[next_state]
                            temp_next_statenumber = np.where(corresponding_next_states[:, 1] == r)[0]
                            assert (len(temp_next_statenumber) > 0)
                            exact_next_state_number = corresponding_next_states[temp_next_statenumber, 0].item()
                            tostoretp = transitions[a]
                            if tostoretp is None:
                                tostoretp = np.array([[cs], [exact_next_state_number], [1.]])
                            else:
                                tostoretp = np.array(tostoretp)
                                tostoretp = np.append(tostoretp, np.array([[cs], [exact_next_state_number], [1.]]), axis=1)
                            transitions[a] = tostoretp
                if r % 100 == 0:
                    np.save(os.path.join(self.borlange_dir, "new_transitions.npy"), transitions)
            np.save(os.path.join(self.borlange_dir, "new_transitions.npy"), transitions)
        else:
            transitions = np.load(os.path.join(self.borlange_dir, "new_transitions.npy"),allow_pickle=True).item()

        #transitions = np.load(os.path.join(self.borlange_dir, "new_transitions.npy"),allow_pickle=True).item()
        nstates,nfeatures = np.shape(feat_data)
        transition_dynamics = {}
        for i in range(self.N_ACTIONS):
            tpsparse = coo_matrix((transitions[i][2, :], (transitions[i][0, :], transitions[i][1, :])),
                         shape=(nstates,nstates))
            tpdense = tpsparse.toarray()
            assert(np.max(np.sum(tpdense,axis=1))==1. and np.min(np.sum(tpdense,axis=1))==1.)
            transition_dynamics[i] = coo_matrix((transitions[i][2, :], (transitions[i][0, :], transitions[i][1, :])),
                         shape=(nstates,nstates))

        self.transition_probability = transition_dynamics
        self.features = feat_data
        self.state_debug = state_debug
        self.nodummy_states = np.array(nodummy_states)



    def evaluate_expsor(self, trajectories):
        gg = self.state_debug[self.destination][:,0].squeeze()

        n_trajectories, l_trajectory, _ = np.shape(trajectories)
        esor = 0.
        discounts = np.array([self.discount**t for t in range(l_trajectory)])
        discounts = np.repeat(np.expand_dims(discounts,axis=0),n_trajectories,0)
        just_states = trajectories[:,:,0]
        just_rewards = self.rewards[just_states]

        """
        trajectories2 = np.array(trajectories)
        for n in range(n_trajectories):
            occured_before = False
            for l in range(l_trajectory):
                if trajectories2[n,l,0] in gg:
                    if occured_before:
                        just_rewards[n,l] = 0.
                    else:
                        occured_before = True
        """
        discounted_states = np.multiply(just_rewards,discounts)
        reward_trajectories = np.sum(discounted_states,axis=1)
        discounted_states = discounted_states.flatten()
        return np.sum(discounted_states)/n_trajectories,reward_trajectories

    def change_format(self, trajectories):
        n_trajectories, l_trajectories = np.shape(trajectories)
        new_trajectories = -2*np.ones((n_trajectories,l_trajectories-1,2))
        #end_pos = []

        print("Cleaning up the expert trajectories")
        for trind,tr in enumerate(trajectories):
            timetostop = False
            for l in np.arange(1,len(tr)-1):
                prev_sno = tr[l-1]
                if prev_sno == -1:
                    prev_sno = tr[l-2]
                    timetostop = True
                current_sno = tr[l]
                if current_sno == -1:
                    current_sno = prev_sno
                next_sno = tr[l + 1]
                if next_sno == -1:
                    next_sno = current_sno
                    #end_pos.append(l+2)

                current_vstates = self.state_debug[current_sno]
                ind = np.where(current_vstates[:,1]==prev_sno)
                current_STATE = current_vstates[ind,0].squeeze().item()

                next_vstates = self.state_debug[next_sno]
                next_ind = np.where(next_vstates[:,1]==current_sno)
                next_STATE = next_vstates[next_ind, 0].squeeze().item()
                current_ACTION = -1
                for a in range(self.N_ACTIONS):
                    rows = self.transition_probability[a].row
                    rowind = np.where(rows == current_STATE)
                    cols = self.transition_probability[a].col
                    poss_next_STATE = cols[rowind]
                    if poss_next_STATE == next_STATE:
                        current_ACTION = a
                        break
                new_trajectories[trind,l-1,0]=current_STATE
                new_trajectories[trind,l-1,1]=current_ACTION
                if timetostop:
                    break

        end_pos = np.argmin(new_trajectories[:,:,0],axis=1)
        max_length = np.amax(end_pos)
        new_trajectories2 = new_trajectories[:,0:max_length,:]
        new_trajectories2 = new_trajectories2.astype(np.int)
        for n in range(n_trajectories):
            new_trajectories2[n,end_pos[n]:,0] = new_trajectories2[n,end_pos[n]-1,0].item()
            new_trajectories2[n,end_pos[n]:,1] = new_trajectories2[n,end_pos[n]-1,1].item()
        return new_trajectories2

    def get_policy(self):
        policy, stoch_policy, v, qvalues = value_iteration(self.transition_probability, self.rewards, self.discount,
                                                           None, self.borlange_dir)
        return stoch_policy,qvalues



