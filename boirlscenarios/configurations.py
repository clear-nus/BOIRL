import os
import boirlscenarios.constants as constants
import numpy as np


class Configurations:
    def __init__(self, kernel, env, projections=None):
        self.kernel = kernel
        self.env = env
        # discount values for various environments
        self.discounts = {constants.GRIDWORLD2D: 0.9,
                          constants.GRIDWORLD3D: 0.9,
                          constants.VIRTBORLANGE: 0.99,
                          constants.REALBORLANGE: 0.99,
                          constants.FETCH: 0.99,
                          constants.MAZE: 0.9
                          }

        # number of artificial trajectories for a given expert trajectory
        # used in calculation of taurbf
        self.nArtTrajs = {constants.GRIDWORLD2D: 5,
                          constants.GRIDWORLD3D: 5,
                          constants.VIRTBORLANGE: 2000,
                          constants.REALBORLANGE: 2000,
                          constants.FETCH: 5,
                          constants.MAZE: 5
                          }

        # number of subset of expert trajectories used for
        self.nTrajs = {constants.GRIDWORLD2D: 10,
                       constants.GRIDWORLD3D: 10,
                       constants.VIRTBORLANGE: 10,
                       constants.REALBORLANGE: 10,
                       constants.FETCH: 20,
                       constants.MAZE: 10
                       }

        # length of each expert trajectory
        self.lTrajs = {constants.GRIDWORLD2D: 15,
                       constants.GRIDWORLD3D: 15,
                       constants.VIRTBORLANGE: 100,
                       constants.REALBORLANGE: 29,
                       constants.FETCH: -1,
                       constants.MAZE: 100
                       }

        if projections is None:
            self.projections = self.getNTrajs()
        else:
            self.projections = projections

        self.plt_xlabels = {constants.GRIDWORLD2D: "Steepness, " + r"$\theta_0$",
                            constants.VIRTBORLANGE: "Time, " + r"$\theta_0$",
                            constants.MAZE: "x-position, " + r"$\theta_0$",
                            constants.FETCH: "Distance Threshold, " + r"$\theta_0$"}
        self.plt_ylabels = {constants.GRIDWORLD2D: "Midpoint, " + r"$\theta_1$",
                            constants.VIRTBORLANGE: "LeftTurn, " + r"$\theta_1$",
                            constants.MAZE: "y-position, " + r"$\theta_1$",
                            constants.FETCH: "Penalty, " + r"$\theta_0$"}
        self.plt_xlims = {constants.GRIDWORLD2D: [-2, 2],
                          constants.VIRTBORLANGE: [-2.5, 0],
                          constants.MAZE: [-1, 1],
                          constants.FETCH: [0., 0.25]}
        self.plt_ylims = {constants.GRIDWORLD2D: [-10, 10],
                          constants.VIRTBORLANGE: [-2.5, 0],
                          constants.MAZE: [-1., 1],
                          constants.FETCH: [-1.5, 1.5]}

        self.plt_xticks = {constants.GRIDWORLD2D: [-2, 0, 2],
                           constants.VIRTBORLANGE: [-2.5, 1.2, 0],
                           constants.MAZE: [-1, 0, 1],
                           constants.FETCH: [0, 0.12, 0.25]}
        self.plt_yticks = {constants.GRIDWORLD2D: [-10, 0, 10],
                           constants.VIRTBORLANGE: [-2.5, 1.2, 0],
                           constants.MAZE: [-1, 0, 1],
                           constants.FETCH: [-1.5, 0., 1.5]}

        self.plt_gt = {constants.GRIDWORLD2D: np.array([[1.25, 5]]),
                       constants.GRIDWORLD3D: np.array([[1.25, 5, 0]]),
                       constants.VIRTBORLANGE: np.array([[-2., -1, -1]]),
                       constants.MAZE: np.array([[0.3, 0.5]]),
                       constants.FETCH: np.array([[0.05, -1]])}

    # directory to store results
    def getResultDir(self):
        # resultdir = os.path.join("ResultDir", self.env, self.kernel, str(self.projections))
        resultdir = os.path.join("ResultDir", self.env, self.kernel)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        return resultdir

    # directory where expert trajectories are saved
    def getTrajectoryDir(self):
        trajdir = os.path.join("Data", self.env)
        if not os.path.exists(trajdir):
            os.makedirs(trajdir)
        return trajdir

    def getDiscounts(self):
        return self.discounts[self.env]

    def getNArtTrajs(self):
        return self.nArtTrajs[self.env]

    def getNTrajs(self):
        return self.nTrajs[self.env]

    def getLTrajs(self):
        return self.lTrajs[self.env]
