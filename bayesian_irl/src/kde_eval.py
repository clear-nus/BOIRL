import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

env = "Gridworld"
ntrials = 10
loaddir = "../Results/%s/"%env
for n in range(ntrials):
    xmin = -2
    xmax = 2
    ymin = -10
    ymax = 10

    fid = os.path.join(loaddir,"rewards%d.npy")%(n)
    weights  = np.load(fid)
    weights = weights.T
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = stats.gaussian_kde(weights)
    dens = np.reshape(kernel(positions).T, X.shape)
    np.save(os.path.join(loaddir,"kde_%d.npy")%n,dens)


    print("sss")
