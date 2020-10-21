import numpy as np
from boirlscenarios.irlobject import IRLObject
import boirlscenarios.constants as constants
import os
import matplotlib.pyplot as plt

def plot_fetch_sr(algos):

    """

    for algo in algos:
        print(algo)
        allsr = []
        irlobj = IRLObject(kernel=algo,env=constants.FETCH)
        eind = 4 if algo == "matern" else 5
        for i in np.arange(1, eind):
            allsr.append(np.load(os.path.join(irlobj.configurations.getResultDir(), "sr%d.npy") % i))
        np.save(os.path.join(irlobj.configurations.getResultDir(),"sr.npy"),np.array(allsr))
    """


    linewd = 3
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    fontlabs = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 30,
                'verticalalignment': 'center',
                'horizontalalignment': 'center'
                }
    fig = plt.figure(11, figsize=(24., 13.5), dpi=80)
    for algo in algos:
        irlobj = IRLObject(kernel=algo,env=constants.FETCH)
        sr = np.load(os.path.join(irlobj.configurations.getResultDir(),"sr.npy"))
        sr_mean = np.mean(sr, axis=0)
        sr_std = np.std(sr, axis=0)
        p1 = plt.fill_between(np.arange(len(sr_mean)), sr_mean - sr_std, sr_mean + sr_std, alpha=0.1)
        plt_color = np.array(p1.get_facecolor()[0])
        plt_color[-1] = 1.
        plt.plot(np.arange(len(sr_mean)), sr_mean, "-", label=constants.LEGENDS[algo], c=plt_color,
                 linewidth=linewd)

    plt.legend(fontsize=20)
    plt.xlabel("Number of iterations", fontdict=fontlabs, labelpad=30)
    plt.ylabel("Success Rate", fontdict=fontlabs, labelpad=15)
    plt.savefig("SR_%s.png" % constants.FETCH, bbox_inches="tight")
    plt.close("all")

    fig = plt.figure(12, figsize=(24., 13.5), dpi=80)
    for algo in algos:
        irlobj = IRLObject(kernel=algo,env=constants.FETCH)
        trials = np.load(os.path.join(irlobj.configurations.getResultDir(), "trial_tracks.npy"))
        Y = np.load(os.path.join(irlobj.configurations.getResultDir(), "y_choices.npy"))
        unique_trial, trcnts = np.unique(trials, return_counts=True)
        unique_trial = np.sort(unique_trial)
        trcnts = np.amax(trcnts)
        algo_lik = None
        for tr in unique_trial:
            # Calculate ESOR and NLL for each trial
            # Find the indices to extract for current trial
            trialInds = np.where(trials == tr)[0]
            """
            if algo_lik is not None:
                if not (len(trialInds) == np.shape(algo_lik)[1]):
                    break
            """
            validY = Y[trialInds]

            # NLL can be directly read off from validY as opposed to other algorithms
            bestY = np.minimum.accumulate(validY)
            bestY = np.append(bestY, bestY[-1].item() * np.ones(trcnts - len(bestY)))
            if algo_lik is None:
                algo_lik = np.expand_dims(bestY, axis=0)
            else:
                algo_lik = np.append(algo_lik, np.expand_dims(bestY, axis=0), axis=0)
        lik_mean = np.mean(algo_lik, axis=0)
        lik_std = np.std(algo_lik, axis=0)
        p1 = plt.fill_between(np.arange(len(lik_mean)), lik_mean - lik_std, lik_mean + lik_std, alpha=0.1)
        plt_color = np.array(p1.get_facecolor()[0])
        plt_color[-1] = 1.
        plt.plot(np.arange(len(lik_mean)), lik_mean, "-", label=constants.LEGENDS[algo], c=plt_color, linewidth=linewd)

    plt.legend(fontsize=20)
    plt.xlabel("Number of iterations", fontdict=fontlabs, labelpad=30)
    plt.ylabel("NLL", fontdict=fontlabs, labelpad=15)
    plt.xticks(np.arange(0, len(lik_mean), 5))
    plt.savefig("NLL_%s.png" % constants.FETCH, bbox_inches="tight")
    plt.close("all")