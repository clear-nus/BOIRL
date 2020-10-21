import numpy as np 

LOAD_FROM_MATLAB_TEXT_FILES = 0
LOAD_FROM_SCIPY_NPZ = 1

travel_time_filename = "mdp/metaborlange/data/ATTRIBUTEestimatedtime.txt"
turn_angle_filename = "mdp/metaborlange/data/ATTRIBUTEturnangles.txt"
incidence_filename = "mdp/metaborlange/data/linkIncidence.txt"
observation_filename = "mdp/metaborlange/data/SyntheticObservations.txt"

npz_path = "data"
load_mode = LOAD_FROM_SCIPY_NPZ

mu = 1.0

regularizer_mean = -3.0
regularizer_std = 1.0

init_beta_func = lambda nfeature: -np.random.rand(nfeature) * 5 - 3e-1
