import numpy as np
import os
from .configurations import Configurations
from GPy.kern.src.rbf import RBF
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
import boirlscenarios.constants as constants
from utils.plotme import plot_me

"""
Main object to store expert trajectories, result directory, trajectory directory etc.
"""


class IRLObject:
    def __init__(self, kernel, env, projections=None):
        self.kernelname = kernel
        self.envname = env
        self.configurations = Configurations(self.kernelname, self.envname, projections)
        self.projections = self.configurations.projections
        self.bounds = None
        self.fullTrajectories = None
        self.fullStartpos = None
        self.trajectories = None
        self.artTrajectories = None
        self.pbar = None
        self.allW = None
        self.gtheta = None
        self.x_choices = []
        self.y_choices = []
        self.stoch_pol = []
        self.trial_track = []
        self.length_track = []
        self.train_fids = []
        self.tr = -1
        self.time_track = []
        self.setupenv()
        self.setupGTheta()
        self.setupkernel()
        self.setupallW()

    def setupenv(self):
        # Generate Environments
        if self.envname == constants.GRIDWORLD2D:
            from mdp.gridworld2d import GridWorld2D
            O = GridWorld2D(horizon=self.configurations.getLTrajs())
        elif self.envname == constants.GRIDWORLD3D:
            from mdp.gridworld3d import GridWorld3D
            O = GridWorld3D(horizon=self.configurations.getLTrajs())
        elif self.envname == constants.VIRTBORLANGE or self.envname == constants.REALBORLANGE:
            from mdp.borlangeworld import BorlangeWorld
            O = BorlangeWorld(destination=7622, horizon=self.configurations.getLTrajs(),
                              discount=self.configurations.getDiscounts(), loadres=True)
        elif self.envname == constants.FETCH:
            from mdp.fetchenv import FetchEnv
            O = FetchEnv()
        elif self.envname == constants.MAZE:
            from mdp.mazeenv import MazeEnv
            O = MazeEnv()
        self.env = O

        # Load trajectories from existing file
        fulltrajloc = os.path.join(self.configurations.getTrajectoryDir(), "full_opt_trajectories.npy")
        if os.path.exists(fulltrajloc):
            self.fullTrajectories = np.load(fulltrajloc)
            self.fullStartpos = np.load(os.path.join(self.configurations.getTrajectoryDir(), "full_start_pos.npy"))
            self.trajectories = np.load(os.path.join(self.configurations.getTrajectoryDir(), "train_trajectories.npy"))
            # self.artTrajectories = np.load(os.path.join(self.configurations.getTrajectoryDir(), "artifical_trajectories.npy"))
        else:
            if self.envname == constants.FETCH or self.envname == constants.MAZE:
                self.fullTrajectories = self.env.trajectories
                self.fullartTrajectories = self.env.art_trajectories
            else:
                raise (FileNotFoundError("No datafound. Have you run datacollect.py?"))

    def setupkernel(self, tr=None, art_tr=None):

        if tr is None:
            tr = self.trajectories
        if art_tr is None:
            art_tr = self.artTrajectories

        # Set up the algorithm with the correct bounds and kernels
        if self.envname == constants.GRIDWORLD2D:
            self.bounds = [{'name': 'var_0', 'type': 'continuous', 'domain': (-2., 2.)},
                           {'name': 'var_1', 'type': 'continuous', 'domain': (-10., 10.)}]
            if self.kernelname == constants.RHORBF:
                from kernel.rhorbfgridworld2d import TauRBF
                self.kernel = TauRBF(input_dim=len(self.bounds), features=self.env.features,
                                     demonstrations=tr, art_demonstrations=art_tr,
                                     discount=self.configurations.getDiscounts())
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.RBF:
                from GPy.kern.src.rbf import RBF
                self.kernel = RBF(len(self.bounds))
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.MATERN:
                from GPy.kern.src.sde_matern import Matern32
                self.kernel = Matern32(len(self.bounds))
        elif self.envname == constants.GRIDWORLD3D:
            self.bounds = [{'name': 'var_0', 'type': 'continuous', 'domain': (-2., 2.)},
                           {'name': 'var_1', 'type': 'continuous', 'domain': (-10., 10.)},
                           {'name': 'var_2', 'type': 'continuous', 'domain': (-4., 4.)}]
            if self.kernelname == constants.RHORBF:
                from kernel.rhorbfgridworld3d import TauRBF
                self.kernel = TauRBF(input_dim=len(self.bounds), features=self.env.features,
                                     demonstrations=tr, art_demonstrations=art_tr,
                                     discount=self.configurations.getDiscounts())
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.RBF:
                from GPy.kern.src.rbf import RBF
                self.kernel = RBF(len(self.bounds))
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.MATERN:
                from GPy.kern.src.sde_matern import Matern32
                self.kernel = Matern32(len(self.bounds))
        elif self.envname == constants.VIRTBORLANGE or self.envname == constants.REALBORLANGE:
            self.bounds = [{'name': 'time_var', 'type': 'continuous', 'domain': (-2.5, 0.0)},
                           {'name': 'left_var', 'type': 'continuous', 'domain': (-2.5, 0.0)},
                           {'name': 'crlks_var', 'type': 'continuous', 'domain': (-2.5, 0.0)}]
            if self.kernelname == constants.RHORBF:
                from kernel.rhorbfborlange import TauRBF
                self.kernel = TauRBF(input_dim=len(self.bounds), features=self.env.features,
                                     demonstrations=tr, art_demonstrations=art_tr,
                                     discount=self.configurations.getDiscounts())  # , lengthscale=0.01)
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.RBF:
                from GPy.kern.src.rbf import RBF
                self.kernel = RBF(len(self.bounds), lengthscale=1)
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.MATERN:
                from GPy.kern.src.sde_matern import Matern32
                self.kernel = Matern32(len(self.bounds), lengthscale=1)
        elif self.envname == constants.FETCH:
            self.bounds = [{'name': 'var_0', 'type': 'continuous', 'domain': (0., 0.25)},
                           {'name': 'var_1', 'type': 'continuous', 'domain': (-1.5, 1.5)}]
            if self.kernelname == constants.RHORBF:
                from kernel.rhorbffetch import TauRBF
                self.kernel = TauRBF(input_dim=len(self.bounds), demonstrations=tr, art_demonstrations=art_tr,
                                     discount=self.configurations.getDiscounts())
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.RBF:
                from GPy.kern.src.rbf import RBF
                self.kernel = RBF(len(self.bounds))
            elif self.kernelname == constants.MATERN:
                from GPy.kern.src.sde_matern import Matern32
                self.kernel = Matern32(len(self.bounds))
        elif self.envname == constants.MAZE:
            self.bounds = [{'name': 'var_0', 'type': 'continuous', 'domain': (-1, 1)},
                           {'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)}]
            if self.kernelname == constants.RHORBF:
                from kernel.rhorbfmaze import TauRBF
                self.kernel = TauRBF(input_dim=len(self.bounds), demonstrations=tr, art_demonstrations=art_tr,
                                     discount=self.configurations.getDiscounts(), lengthscale=0.6)
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.RBF:
                from GPy.kern.src.rbf import RBF
                self.kernel = RBF(len(self.bounds), lengthscale=1)
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            elif self.kernelname == constants.MATERN:
                from GPy.kern.src.sde_matern import Matern32
                self.kernel = Matern32(len(self.bounds))
                self.kernel.lengthscale.constrain_bounded(lower=1e-36, upper=10)
            # self.kernel.lengthscale.fix()

    # The negative log likelihood function that the BO is trying to optimize
    def blackboxfunc(self, x):
        self.length_track.append(self.kernel.lengthscale.values.item())
        self.pbar.update(1)
        self.x_choices.append(x)
        assert (len(x) == 1)
        self.env.set_reward(x[0])
        if self.envname == constants.FETCH:
            y, fids = self.env.get_likelihood(self.fullTrajectories)
            self.train_fids.append(fids)
        elif self.envname == constants.MAZE:
            self.env.trial = len(self.x_choices)
            self.env.algo = self.kernelname
            y = self.env.get_likelihood()
        elif self.envname == constants.VIRTBORLANGE or self.envname == constants.REALBORLANGE or self.envname == constants.GRIDWORLD2D or self.envname == constants.GRIDWORLD3D:
            y, stochpol = self.env.get_likelihood(self.fullTrajectories)
            self.stoch_pol.append(stochpol)
        else:
            y = self.env.get_likelihood(self.fullTrajectories)
        self.y_choices.append(y)
        self.trial_track.append(self.tr)
        # np.save(os.path.join(self.configurations.getResultDir(), "x_choices.npy"), np.array(self.x_choices))
        # np.save(os.path.join(self.configurations.getResultDir(), "y_choices.npy"), np.array(self.y_choices))
        # np.save(os.path.join(self.configurations.getResultDir(), "trial_tracks.npy"), np.array(self.trial_track))
        return y

    # Plot posterior mean and std
    def plotAcquisition(self, bo, trial):
        fontlabs = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 50,
                    'verticalalignment': 'center',
                    'horizontalalignment': 'center'
                    }

        bounds = bo.acquisition.space.get_bounds()
        canplot = False
        model = bo.model
        input_dim = model.model.X.shape[1]
        Xdata = model.model.X
        ns = 200

        # Plot
        if len(bounds) == 2:
            canplot = True
            ns = 200
            X1 = np.linspace(bounds[0][0], bounds[0][1], ns)
            X2 = np.linspace(bounds[1][0], bounds[1][1], ns)
            x1, x2 = np.meshgrid(X1, X2)
            X = np.hstack((x1.reshape(ns * ns, 1), x2.reshape(ns * ns, 1)))
            if self.envname == constants.VIRTBORLANGE:
                X = np.hstack((X, -1 * np.ones((X.shape[0], 1))))
                X = np.hstack((X, -20 * np.ones((X.shape[0], 1))))
            m, v = model.predict(X)
        if len(bounds) == 3:
            canplot = False
            if self.envname == constants.VIRTBORLANGE:
                canplot = True
                ns = 50
                X1 = np.linspace(bounds[0][0], bounds[0][1], ns)
                X2 = np.linspace(bounds[1][0], bounds[1][1], ns)
                X3 = -1 * np.ones((ns * ns, 1))
                x1, x2 = np.meshgrid(X1, X2)
                X = np.hstack((x1.reshape(ns * ns, 1), x2.reshape(ns * ns, 1)))
                X = np.hstack((X, X3))
            else:
                ns = 13
                X1 = np.linspace(bounds[0][0], bounds[0][1], ns)
                X2 = np.linspace(bounds[1][0], bounds[1][1], ns)
                X3 = np.linspace(bounds[2][0], bounds[2][1], ns)
                x1, x2, x3 = np.meshgrid(X1, X2, X3)
                X = np.hstack((x1.reshape(ns * ns * ns, 1), x2.reshape(ns * ns * ns, 1), x3.reshape(ns * ns * ns, 1)))
            m = np.zeros((X.shape[0], 1))
            v = np.zeros((X.shape[0], 1))
            if self.envname == constants.VIRTBORLANGE:
                for myxind, myx in enumerate(X):
                    myx = np.expand_dims(myx, axis=0)
                    mym, myv = model.predict(myx)
                    m[myxind] = mym
                    v[myxind] = myv
            else:
                m, v = model.predict(X)

            # m, v = model.predict(X)
            np.save(os.path.join(self.configurations.getResultDir(), "X3%d.npy") % trial, X3)

        if len(bounds) == 4 and not (self.envname == constants.VIRTBORLANGE):
            canplot = False
            ns = 20
            X1 = np.linspace(bounds[0][0], bounds[0][1], ns)
            X2 = np.linspace(bounds[1][0], bounds[1][1], ns)
            X3 = np.linspace(bounds[2][0], bounds[2][1], ns)
            X4 = np.linspace(bounds[3][0], bounds[3][1], ns)
            x1, x2, x3, x4 = np.meshgrid(X1, X2, X3, X4)
            X = np.hstack(
                (x1.reshape(ns * ns * ns * ns, 1), x2.reshape(ns * ns * ns * ns, 1), x3.reshape(ns * ns * ns * ns, 1),
                 x4.reshape(ns * ns * ns * ns, 1)))
            m, v = model.predict(X)
            np.save(os.path.join(self.configurations.getResultDir(), "X3%d.npy") % trial, X3)
            np.save(os.path.join(self.configurations.getResultDir(), "X4%d.npy") % trial, X4)

        np.save(os.path.join(self.configurations.getResultDir(), "X1%d.npy") % trial, X1)
        np.save(os.path.join(self.configurations.getResultDir(), "X2%d.npy") % trial, X2)
        np.save(os.path.join(self.configurations.getResultDir(), "m%d.npy") % trial, m)
        np.save(os.path.join(self.configurations.getResultDir(), "v%d.npy") % trial, v)
        np.save(os.path.join(self.configurations.getResultDir(), "Xdata%d.npy") % trial, Xdata)
        if canplot:
            plt.close("all")
            plot_me(pos=X, env=self.envname, algo=self.kernelname, val=m, notgreat=None, goodish=Xdata, best=None,
                    is_ours=True, savedir=self.configurations.getResultDir(), fname="PosteriorMean%d.png" % trial,
                    plt_xlabels=self.configurations.plt_xlabels[self.envname],
                    plt_ylabels=self.configurations.plt_ylabels[self.envname],
                    plt_xticks=self.configurations.plt_xticks[self.envname],
                    plt_yticks=self.configurations.plt_yticks[self.envname],
                    plt_xlims=self.configurations.plt_xlims[self.envname],
                    plt_ylims=self.configurations.plt_ylims[self.envname],
                    plt_gt=self.configurations.plt_gt[self.envname], trial=trial, ismean=True)

            plt.close("all")
            plot_me(pos=X, env=self.envname, algo=self.kernelname, val=v, notgreat=None, goodish=Xdata, best=None,
                    is_ours=True, savedir=self.configurations.getResultDir(), fname="PosteriorStd%d.png" % trial,
                    plt_xlabels=self.configurations.plt_xlabels[self.envname],
                    plt_ylabels=self.configurations.plt_ylabels[self.envname],
                    plt_xticks=self.configurations.plt_xticks[self.envname],
                    plt_yticks=self.configurations.plt_yticks[self.envname],
                    plt_xlims=self.configurations.plt_xlims[self.envname],
                    plt_ylims=self.configurations.plt_ylims[self.envname],
                    plt_gt=self.configurations.plt_gt[self.envname], trial=trial, ismean=True)

    # Store random reward function parameters within bounds
    def setupallW(self):
        if os.path.exists(os.path.join(self.configurations.getTrajectoryDir(), "allw.npy")):
            self.allW = np.load(os.path.join(self.configurations.getTrajectoryDir(), "allw.npy"))
        else:
            self.allW = None

    # Store ground truth reward function parameters
    def setupGTheta(self):
        if self.envname == constants.REALBORLANGE:
            self.gtheta = np.zeros((1,3))
        else:
            self.gtheta = self.configurations.plt_gt[self.envname]
