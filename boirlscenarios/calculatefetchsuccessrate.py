import numpy as np
import matplotlib.pyplot as plt
from mdp.fetchenv import FetchEnv
from boirlscenarios.irlobject import IRLObject
import os
from tqdm import tqdm
import boirlscenarios.constants as constants
from mdp.metafetch.fetchgeneratetrajs import generate_trajectories

folder = "logs/"

def evalsuccessrate(algo, env, projections=None):
    assert(env == constants.FETCH)
    algo_lik = None
    algo_sr = None
    boirlobj = IRLObject(algo, env, projections)
    gt_trajs = boirlobj.fullTrajectories
    gt_spos = boirlobj.fullStartpos
    trials = np.load(os.path.join(boirlobj.configurations.getResultDir(), "trial_tracks.npy"))
    X = np.load(os.path.join(boirlobj.configurations.getResultDir(), "x_choices.npy"))
    Y = np.load(os.path.join(boirlobj.configurations.getResultDir(), "y_choices.npy"))
    policies = None
    policy_location = os.path.join(boirlobj.configurations.getResultDir(), "stochpols.npy")
    if os.path.exists(policy_location):
        policies = np.load(policy_location)
    unique_trial, trcnts = np.unique(trials, return_counts=True)
    unique_trial = np.sort(unique_trial)
    trcnts = np.amax(trcnts)

    for tr in tqdm(unique_trial):
        # Calculate ESOR and NLL for each trial
        # Find the indices to extract for current trial
        trialInds = np.where(trials == tr)[0]
        """
        if algo_lik is not None:
            if not (len(trialInds) == np.shape(algo_lik)[1]):
                break
        """
        validX = X[trialInds]
        validY = Y[trialInds]
        validP = None
        if policies is not None:
            validP = policies[trialInds]

        # NLL can be directly read off from validY as opposed to other algorithms
        bestY = np.minimum.accumulate(validY)
        bestY = np.append(bestY, bestY[-1].item() * np.ones(trcnts - len(bestY)))
        if algo_lik is None:
            algo_lik = np.expand_dims(bestY, axis=0)
        else:
            algo_lik = np.append(algo_lik, np.expand_dims(bestY, axis=0), axis=0)

        # Calculate Success Rate
        validChangeLoc = bestY[1:] - bestY[0:-1]
        validChangeLoc = np.append(np.ones(1), validChangeLoc)
        temp_sr = np.zeros(len(bestY))
        for n in range(len(bestY)):
            if not (validChangeLoc[n] == 0):
                current_x = validX[n]

                # Set the reward function to the learned reward function
                boirlobj.env.set_reward(current_x.squeeze())

                # get current policy
                current_policy = None
                if validP is not None:
                    current_policy = validP[n]

                randomstart = False
                if env == constants.FETCH:
                    n_testtrajs = gt_trajs["observation"].shape[0]
                else:
                    n_testtrajs = gt_trajs.shape[0]
                spos = None

                _, fid = boirlobj.env.get_likelihood(trajectories=None)

                for currentfid in fid:
                    _, current_sr, current_mr = generate_trajectories("FetchReach-v1", "her", folder, currentfid)
            temp_sr[n] = np.mean(current_sr)
        if algo_sr is None:
            algo_sr = np.expand_dims(temp_sr, axis=0)
        else:
            algo_sr = np.append(algo_sr, np.expand_dims(temp_sr, axis=0), axis=0)

    # Save ESOR and NLL
    np.save(os.path.join(boirlobj.configurations.getResultDir(), "sr.npy"), algo_sr)
    np.save(os.path.join(boirlobj.configurations.getResultDir(), "likelihood.npy"), algo_lik)

