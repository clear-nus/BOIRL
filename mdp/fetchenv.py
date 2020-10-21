import numpy as np
import os
from mdp.metafetch.likelihood import likelihood


class FetchEnv:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    def __init__(self):
        self.metadir = "mdp/metafetch"
        self.indices = np.load(os.path.join(self.metadir, "fulldone.npy"))
        self.indices[0] = 1
        self.actions = np.load(os.path.join(self.metadir, "fullactions.npy")).squeeze()

        self.trajectories = {}
        self.trajectories["observation"] = np.load(os.path.join(self.metadir, "fullobs.npy")).squeeze()
        self.trajectories["achieved_goal"] = np.load(os.path.join(self.metadir, "fullachieved.npy")).squeeze()
        self.trajectories["desired_goal"] = np.load(os.path.join(self.metadir, "fulldesired.npy")).squeeze()
        self.trajectories["indices"] = self.indices
        self.trajectories["actions"] = self.actions

        self.art_trajectories = {}
        self.art_trajectories["observation"] = np.load(os.path.join(self.metadir, "art_obs.npy")).squeeze()
        self.art_trajectories["desired_goal"] = np.load(os.path.join(self.metadir, "art_desired.npy")).squeeze()
        self.art_trajectories["achieved_goal"] = np.load(os.path.join(self.metadir, "art_achieved.npy")).squeeze()
        self.art_trajectories["actions"] = np.load(os.path.join(self.metadir, "art_actions.npy")).squeeze()

        self.NTRAJ, self.LTRAJ = self.get_maxL()
        self.weights = np.array([0.05, -1.])

    def set_reward(self, x):
        self.weights = x

    def get_likelihood(self, trajectories):
        ll = 0.
        fids = []
        for i in range(3):
            currentll, current_fid = likelihood(self.weights, self.trajectories)
            ll += currentll
            fids.append(current_fid)
        return ll/3, fids

    def get_subs(self, nsublens):
        inds = np.where(self.indices == 1.)[0]
        subinds = np.random.permutation(inds.shape[0])[0:nsublens]
        inds = np.append(inds, self.indices.shape[0]).astype(np.int)
        assert (len(inds) >= nsublens)
        subsetTrajectories = {}
        subsetArtTrajectories = {}

        tempObs = None
        tempAch = None
        tempDes = None
        tempIndices = None
        tempActions = None

        tempArtObs = None
        tempArtDes = None
        tempArtAchieved = None
        tempArtActions = None

        for s in subinds:
            if tempObs is None:
                tempObs = self.trajectories["observation"][inds[s]:inds[s + 1]]
                tempAch = self.trajectories["achieved_goal"][inds[s]:inds[s + 1]]
                tempDes = self.trajectories["desired_goal"][inds[s]:inds[s + 1]]
                tempIndices = self.trajectories["indices"][inds[s]:inds[s + 1]]
                tempActions = self.trajectories["actions"][inds[s]:inds[s + 1]]
                tempArtObs = self.art_trajectories["observation"][:, inds[s]:inds[s + 1]]
                tempArtDes = self.art_trajectories["desired_goal"][:, inds[s]:inds[s + 1]]
                tempArtAchieved = self.art_trajectories["achieved_goal"][:, inds[s]:inds[s + 1]]
                tempArtActions = self.art_trajectories["actions"][:, inds[s]:inds[s + 1]]
            else:
                tempObs = np.append(tempObs, self.trajectories["observation"][inds[s]:inds[s + 1]], axis=0)
                tempAch = np.append(tempAch, self.trajectories["achieved_goal"][inds[s]:inds[s + 1]], axis=0)
                tempDes = np.append(tempDes, self.trajectories["desired_goal"][inds[s]:inds[s + 1]], axis=0)
                tempIndices = np.append(tempIndices, self.trajectories["indices"][inds[s]:inds[s + 1]])
                tempActions = np.append(tempActions, self.trajectories["actions"][inds[s]:inds[s + 1]], axis=0)
                tempArtObs = np.append(tempArtObs, self.art_trajectories["observation"][:, inds[s]:inds[s + 1]], axis=1)
                tempArtDes = np.append(tempArtDes, self.art_trajectories["desired_goal"][:, inds[s]:inds[s + 1]],
                                       axis=1)
                tempArtAchieved = np.append(tempArtAchieved, self.art_trajectories["achieved_goal"][:, inds[s]:inds[s + 1]],
                                       axis=1)
                tempArtActions = np.append(tempArtActions, self.art_trajectories["actions"][:, inds[s]:inds[s + 1]],
                                           axis=1)

        subsetTrajectories["observation"] = tempObs
        subsetTrajectories["achieved_goal"] = tempAch
        subsetTrajectories["desired_goal"] = tempDes
        subsetTrajectories["indices"] = tempIndices
        subsetTrajectories["actions"] = tempActions

        subsetArtTrajectories["observation"] = tempArtObs
        subsetArtTrajectories["desired_goal"] = tempArtDes
        subsetArtTrajectories["actions"] = tempArtActions
        subsetArtTrajectories["achieved_goal"] = tempArtAchieved

        return subsetTrajectories, None, subsetArtTrajectories


    def get_maxL(self):
        inds = np.where(self.indices == 1.)[0]
        inds = np.append(inds, self.indices.shape[0]).astype(np.int)
        leng = inds[1:] - inds[0:-1]
        return len(inds) - 1, np.amax(leng)

    def generate_trajectories(self,n_trajectories=None):
        raise(NotImplementedError)
