import numpy as np 
import scipy.sparse as sparse
import mdp.metaborlange.utils as utils
import pickle


class Routedata():

    def __init__(self):

        self.incidence_smat_no_dummy = None 
        self.incidence_smat_dummy = None 
        self.turn_angle_smat = None 
        self.observation_smat = None 

        self.incidence_smat = None
        self.lefturn_smat = None
        self.uturn_smat = None
        self.travel_time_smat = None


    def load_incidence_data(self, incidence_filename):
        incidence_data = np.loadtxt(incidence_filename)
        self.incidence_smat  = utils.convert_to_sparse_mat(incidence_data)
        self.incidence_smat_square = utils.convert_to_sparse_mat(incidence_data,square=True,dim=1)
        # 7288x7754
        # 7288x7288: incidence matrix of links in the network
        # [:,7288:7754]: incidence matrix for the dummy link, w.r.t. the observation file

        self.nlink = self.incidence_smat.shape[0]

        self.incidence_smat_no_dummy = self.incidence_smat[:,:self.nlink]
        self.incidence_smat_dummy = self.incidence_smat[:,self.nlink:]

        return self.incidence_smat_no_dummy, self.incidence_smat_dummy


    def get_adjlist(self):
        """
        adjlist as a matrix of size nlink x maximum_number_of_outgoing_connections
        each row starts with number of connections connected to that row idx link
            then the link idx
            zero are padding at the end
        """
        assert self.incidence_smat_no_dummy is not None, "incidence matrix must be loaded"
        assert self.incidence_smat_no_dummy.format == 'csr', "incidence matrix must be in CSR format to efficiently generate adjacency list"

        nlink = self.incidence_smat_no_dummy.shape[0]
        n_connect = np.array(self.incidence_smat_no_dummy.sum(axis = 1)).reshape(-1,).astype(int)
        n_max = np.max( n_connect )

        self.adjlist = np.zeros([nlink, n_max+1]).astype(int) 
        # n_max + 1 coz the first element is the number of connections for that row-idx link

        self.adjlist[:,0] = n_connect

        for i in range(nlink):
            start = self.incidence_smat_no_dummy.indptr[i]
            end = self.incidence_smat_no_dummy.indptr[i+1]
            n_connect_i = n_connect[i]
            # assert end - start == n_connect_i, "{}. {} and {} are different".format(i, end - start, n_connect_i)
            self.adjlist[i,1:(n_connect_i+1)] = self.incidence_smat_no_dummy.indices[start:end]

        return self.adjlist


    def load_travel_time_data(self, travel_time_filename):
        assert self.incidence_smat is not None, "incidence data must be loaded!"

        # Load data
        travel_time_data = np.loadtxt(travel_time_filename)
        self.travel_time_smat  = utils.convert_to_sparse_mat(travel_time_data, square = True, dim = 0)
        # 7288x7289

        self.travel_time_smat = utils.make_feature_smat_same_csr_format_as_incidence_smat(
                                self.travel_time_smat, 
                                self.incidence_smat_no_dummy)

        return self.travel_time_smat


    def load_turn_angle_data(self, turn_angle_filename):
        turn_angle_data = np.loadtxt(turn_angle_filename)
        self.turn_angle_smat  = utils.convert_to_sparse_mat(turn_angle_data, square = True, dim = 0)
        # 7288x7287

        return self.turn_angle_smat


    def get_uturn_feature_from_turn_angle(self):
        assert self.turn_angle_smat is not None, "turn angle data must be loaded!"
        assert self.incidence_smat is not None, "incidence data must be loaded!"

        # U-turns: np.abs(turn_angle) > 3.1
        uturn_data = np.zeros(self.turn_angle_smat.data.shape)
        uturn_data[ np.abs(self.turn_angle_smat.data) > 3.1 ] = 1.0

        self.uturn_smat = sparse.csr_matrix((uturn_data,
            self.turn_angle_smat.indices, self.turn_angle_smat.indptr), 
            shape = self.turn_angle_smat.shape)

        self.uturn_smat = utils.make_feature_smat_same_csr_format_as_incidence_smat(
                                self.uturn_smat, 
                                self.incidence_smat_no_dummy)

        return self.uturn_smat
    

    def get_lefturn_feature_from_turn_angle(self):
        assert self.turn_angle_smat is not None, "turn angle data must be loaded!"
        assert self.incidence_smat is not None, "incidence data must be loaded!"

        # Left turn: -3.1 < turn_angle < -0.5236
        lefturn_data = np.zeros(self.turn_angle_smat.data.shape)
        lefturn_data[ (self.turn_angle_smat.data > -3.1) * (self.turn_angle_smat.data < -0.5236)] = 1.0

        self.lefturn_smat = sparse.csr_matrix((lefturn_data,
            self.turn_angle_smat.indices, self.turn_angle_smat.indptr),
            shape = self.turn_angle_smat.shape)

        self.lefturn_smat = utils.make_feature_smat_same_csr_format_as_incidence_smat(
                                self.lefturn_smat, 
                                self.incidence_smat_no_dummy)

        return self.lefturn_smat


    def check_travel_time(self):
        assert self.travel_time_smat is not None, 'Travel time data must be loaded!'
        assert self.incidence_smat_no_dummy is not None, 'Incidence data must be loaded!'

        if np.prod(self.travel_time_smat[ self.incidence_smat_no_dummy > 0.0 ]) < 1e-9:
            print("No. of positive-travel-time: {}".format(self.travel_time_smat.data.shape))
            
            travel_time_for_valid_connection = self.travel_time_smat[ self.incidence_smat_no_dummy > 0.0 ]
            zero_travel_time_for_valid_connection = \
                travel_time_for_valid_connection[travel_time_for_valid_connection == 0.0]
            positive_travel_time_for_valid_connection = \
                travel_time_for_valid_connection[travel_time_for_valid_connection > 0.0]

            print("No. of zero-travel-time link pairs: {}".format(
                zero_travel_time_for_valid_connection.shape))
            print("No. of positive-travel-time link pairs: {}".format(
                positive_travel_time_for_valid_connection.shape))

            if zero_travel_time_for_valid_connection.shape[1] > 0:
                print('** Warning: Some connections (consecutive link pairs) have zero travel time!')
                return False
            else:
                print('+ Pass checking travel time for all connections to be positive.')
                return True


    def load_observation_full_data(self, observation_full_data_filename):
        observation_data = np.loadtxt(observation_full_data_filename)
        self.observation_smat  = utils.convert_to_sparse_mat(observation_data)
        # 1832x93

        return self.observation_smat


    def check_observation_data(self):
        """
        Verifies:
            the last link of each trajectory (the last nonzero number in each row) 
            is the dummy link index in the incidence matrix
        Notes:
            observation indices are 1-based, not 0-based
            observation matrix:
                each row is a trajectory starting with the dummy destination
                    ending with the dummy destination
        """
        assert self.observation_smat is not None, 'Observation data must be loaded!'
        assert self.incidence_smat_dummy is not None, 'Incidence data must be loaded!'

        fail_check = False

        for i in range(self.observation_smat.shape[0]):
            if self.observation_smat.indptr[i+1] - self.observation_smat.indptr[i] > 0:
                # observed trajectory length > 0
                dummy_dest = int(self.observation_smat.data[ self.observation_smat.indptr[i+1]-1 ]) - 1
                dest = int(self.observation_smat.data[ self.observation_smat.indptr[i+1]-2 ]) - 1

                if dummy_dest < self.nlink:
                    fail_check = True
                    print("Some observed trajectory does not end with a dummy link!")
                else:
                    if self.incidence_smat_dummy[dest, dummy_dest - self.nlink] == 0.0:
                        fail_check = True
                        print("Destination is not connected to the correct dummy dest!")
                    # haven't check if incidence_smat[dest, dummy_dest] == 1.0 for some
                    #   dest not connected to dummy_dest

        if not fail_check:
            print("+ Pass checking observation file, and incidence matrix dummy link.")

        return fail_check

