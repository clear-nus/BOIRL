import numpy as np
import matplotlib.pyplot as plt
import os
import boirlscenarios.constants as constants
from scipy import stats

plt_xlabels = {constants.GRIDWORLD2D: "Steepness, " + r"$\theta_0$",
               constants.VIRTBORLANGE: "Time, " + r"$\theta_0$",
               constants.MAZE: "x-position, " + r"$\theta_0$",
               constants.FETCH: "Distance Threshold, " + r"$\theta_0$"}
plt_ylabels = {constants.GRIDWORLD2D: "Midpoint, " + r"$\theta_1$",
               constants.VIRTBORLANGE: "LeftTurn, " + r"$\theta_1$",
               constants.MAZE: "y-position, " + r"$\theta_1$",
               constants.FETCH: "Penalty, " + r"$\theta_0$"}
plt_xlims = {constants.GRIDWORLD2D: [-2, 2],
             constants.VIRTBORLANGE: [-2.5, 0],
             constants.MAZE: [-1, 1],
             constants.FETCH: [0., 0.25]}
plt_ylims = {constants.GRIDWORLD2D: [-10, 10],
             constants.VIRTBORLANGE: [-2.5, 0],
             constants.MAZE: [-1., 1],
             constants.FETCH: [-1, 1]}

plt_xticks = {constants.GRIDWORLD2D: [-2, 0, 2],
              constants.VIRTBORLANGE: [-2.5, 1.2, 0],
              constants.MAZE: [-1, 0, 1],
              constants.FETCH: [0, 0.12, 0.25]}
plt_yticks = {constants.GRIDWORLD2D: [-10, 0, 10],
              constants.VIRTBORLANGE: [-2.5, 1.2, 0],
              constants.MAZE: [-1, 0, 1],
              constants.FETCH: [-1, 0., 1]}

plt_gt = {constants.GRIDWORLD2D: np.array([[1.25, 5]]),
          constants.VIRTBORLANGE: np.array([[-2., -1]]),
          constants.MAZE: np.array([[0.3, 0.5]]),
          constants.FETCH: np.array([[0.05, -1]])}

baseDir = "PosteriorPlotsFromPaper"
resultsDir = os.path.join(baseDir, "Data")


def prep_data(loaddir, trial, is_ours=True, should_truncate=False):
    if is_ours:
        # Load X and Y positions
        X1 = np.load(os.path.join(loaddir, "X1%d.npy") % trial)
        X2 = np.load(os.path.join(loaddir, "X2%d.npy") % trial)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape((len(X1) * len(X1), 1)), x2.reshape((len(X1) * len(X1), 1))))

        # Load the m
        m = np.load(os.path.join(loaddir, "m%d.npy") % trial).squeeze()

        # Load the v
        v = np.load(os.path.join(loaddir, "v%d.npy") % trial).squeeze()

        # Load the ground truth
        gt_lik = np.load(os.path.join(loaddir, "gt_lik.npy"), allow_pickle=True).tolist()

        # Load learned x and y
        x_choices = np.load(os.path.join(loaddir, "x_choices.npy")).squeeze()
        y_choices = np.load(os.path.join(loaddir, "y_choices.npy"))

        # Find those corresponding to the current trial under investigation
        trialtracks = np.load(os.path.join(loaddir, "trial_tracks.npy"))
        trinds = np.where(trialtracks == trial)[0]
        bo_valid_x = x_choices[trinds]
        bo_valid_y = y_choices[trinds]

        #
        if should_truncate:
            trunc_ind = np.where(X[:, 0] < -0.6)[0]
            newX = X[trunc_ind]
            newm = m[trunc_ind]
            newv = v[trunc_ind]
            trunc_ind_bodata = np.where(bo_valid_x[:, 0] < -0.6)[0]
            bo_valid_x = bo_valid_x[trunc_ind_bodata]
            bo_valid_y = bo_valid_y[trunc_ind_bodata]
            xlims = plt_xlims[env]
            xlims[1] = -0.6
            plt_xlims[env] = xlims
            xticks = plt_xticks[env]
            xticks[2] = -0.6
            plt_xticks[env] = xticks
        else:
            newX = X
            newm = m
            newv = v

        broadnearInds = np.where(bo_valid_y <= 1.1 * gt_lik)[0]
        broadfarInds = np.where(bo_valid_y > 1.01 * gt_lik)[0]
        nearInds = np.where(bo_valid_y <= 1.01 * gt_lik)[0]
        farInds = np.where(bo_valid_y >= gt_lik)[0]

        notgreatInds = np.intersect1d(broadnearInds, broadfarInds)
        goodishInds = np.intersect1d(nearInds, farInds)
        bestInds = np.where(bo_valid_y < gt_lik)[0]

        bo_notgreat_xdata = bo_valid_x[notgreatInds]
        bo_goodish_xdata = bo_valid_x[goodishInds]
        bo_best_xdata = bo_valid_x[bestInds]
        bo_notgreat_ydata = bo_valid_y[notgreatInds].squeeze()
        bo_goodish_ydata = bo_valid_y[goodishInds].squeeze()
        bo_best_ydata = bo_valid_y[bestInds].squeeze()


    else:
        xmin = plt_xlims[env][0]
        xmax = plt_xlims[env][1]
        ymin = plt_ylims[env][0]
        ymax = plt_ylims[env][1]
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        newX = np.vstack([X.ravel(), Y.ravel()])
        weights = np.load(os.path.join(loaddir, "rewards%d.npy") % trial)
        weights = weights.T
        kernel = stats.gaussian_kde(weights)
        newm = kernel(newX).T
        newv = None
        newX = newX.T
        bo_goodish_xdata = None
        bo_best_xdata = None
        bo_notgreat_xdata = None

    return newX, newm, newv, bo_notgreat_xdata, bo_goodish_xdata, bo_best_xdata,


def plot_me(pos, env, algo, val, notgreat, goodish, best, is_ours, trial=0, ismean=True):
    markersize = 1000
    savedir = os.path.join(baseDir, "Plots", env)
    if not is_ours:
        savedir = os.path.join(savedir, "BIRL")
    os.makedirs(savedir, exist_ok=True)
    fontlabs = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 50,
                'verticalalignment': 'center',
                'horizontalalignment': 'center'
                }

    plt.close("all")
    plt.figure(1, figsize=(16.75, 13.5), dpi=80)
    if is_ours:
        cmap = "viridis"
    else:
        cmap = "viridis_r"
    p1 = plt.tricontourf(pos[:, 0], pos[:, 1], val.squeeze(), 30, vmin=np.amin(val.squeeze()),
                         vmax=np.amax(val.squeeze()), cmap=cmap)

    plt.xlabel(plt_xlabels[env], fontdict=fontlabs, labelpad=30)
    plt.ylabel(plt_ylabels[env], fontdict=fontlabs, labelpad=15)

    plt.xticks(plt_xticks[env], fontsize=30)
    plt.yticks(plt_yticks[env], fontsize=30)

    cbar = plt.colorbar(p1, ticks=[np.amax(val), np.amin(val), (np.amax(val) + np.amin(val)) / 2.], format='%.1f')
    cbar.ax.tick_params(labelsize=30)
    if env == constants.GRIDWORLD2D:
        if is_ours:
            clr = "white"
        else:
            clr = "black"
        plt.annotate(r"$\theta_2$=0", (-1.8, -9), fontsize=40, c=clr)
    elif env == constants.VIRTBORLANGE:
        plt.annotate(r"$\theta_2$=-1", (-2.4, -2.4), fontsize=40, c="black")
    plt.axis((plt_xlims[env][0], plt_xlims[env][1], plt_ylims[env][0], plt_ylims[env][1]))
    if notgreat is not None:
        plt.scatter(notgreat[:, 0], notgreat[:, 1], edgecolors="r", marker='o', facecolors="none", linewidths=3,
                    s=markersize, zorder=2)
    if goodish is not None:
        plt.scatter(goodish[:, 0], goodish[:, 1], c='r', s=markersize, zorder=2)
    if best is not None:
        plt.scatter(best[:, 0], best[:, 1], marker="X", c="red", s=markersize, zorder=2)
    if not (env == constants.FETCH):
        plt.scatter(plt_gt[env][:, 0], plt_gt[env][:, 1], marker="*", c="lime", s=1500, zorder=1)

    fname = "Posterior_BIRL_%d.png" % trial if not is_ours else (
        "Posterior_mean_%s_%d.png" % (algo, trial) if ismean else "Posterior_std_%s_%d.png" % (algo, trial))

    plt.savefig(os.path.join(savedir, fname), bbox_inches="tight")
    plt.close("all")


def execute_plot(env, algo, load_dir, trial, should_truncate=False, is_ours=True):
    pos, val, val_dev, notgreat, goodish, best = prep_data(loaddir=load_dir, trial=trial, is_ours=is_ours,
                                                           should_truncate=should_truncate)
    plot_me(pos=pos, env=env, algo=algo, val=val, notgreat=notgreat, goodish=goodish, best=best, is_ours=is_ours,
            trial=trial)
    if val_dev is not None:
        plot_me(pos=pos, env=env, algo=algo, val=val_dev, notgreat=notgreat, goodish=goodish, best=best, is_ours=is_ours,
            trial=trial, ismean=False)


## Gridworld3d
env = constants.GRIDWORLD2D
algo = "rhorbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)
algo = "rbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)
algo = "matern"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)

## Vborlange
env = constants.VIRTBORLANGE
algo = "rhorbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=0, should_truncate=True)
algo = "rbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=0, should_truncate=True)
algo = "matern"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=0, should_truncate=True)

## MAZE
env = constants.MAZE
algo = "rhorbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=1)
algo = "rbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=1)
algo = "matern"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=1)

## FETCH
env = constants.FETCH
algo = "rhorbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)
algo = "rbf"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)
algo = "matern"
execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, env, "%s")%algo, trial=3)


algo="birl"
# BIRL
env = constants.GRIDWORLD2D
for i in range(10):
    execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, "BayesianIRL", env), trial=i, is_ours=False)

# BIRL
env = constants.VIRTBORLANGE
for i in range(10):
    execute_plot(env=env, algo=algo, load_dir=os.path.join(resultsDir, "BayesianIRL", env), trial=i, is_ours=False)
