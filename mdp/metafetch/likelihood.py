import numpy as np
from mdp.metafetch.trainFetch2 import train
from mdp.metafetch.liklihoodFetch import getlikelihood
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def likelihood(weights, trajectories=None,current_fid = None):
    if trajectories is None:
        trajectories = {}
        trajectories["observation"] = np.load("fullobs.npy").squeeze()
        trajectories["achieved_goal"] = np.load("fullachieved.npy").squeeze()
        trajectories["desired_goal"] = np.load("fulldesired.npy").squeeze()

    envid = "FetchReach-v1"
    algo = "her"
    folder = "logs/"
    current_fid = train(envid,algo,weights)
    #print(current_fid)
    ll = getlikelihood(envid, algo, folder, current_fid, trajectories)
    return ll, current_fid

