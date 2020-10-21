import numpy as np
import matplotlib.pyplot as plt
import os
import boirlscenarios.constants as constants


def plot_me(pos, env, algo, val, notgreat, goodish, best, is_ours, savedir, fname, plt_xlabels, plt_ylabels, plt_xticks,
            plt_yticks, plt_xlims, plt_ylims, plt_gt, trial=0, ismean=True):
    markersize = 1000

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

    plt.xlabel(plt_xlabels, fontdict=fontlabs, labelpad=30)
    plt.ylabel(plt_ylabels, fontdict=fontlabs, labelpad=15)

    plt.xticks(plt_xticks, fontsize=30)
    plt.yticks(plt_yticks, fontsize=30)

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
    plt.axis((plt_xlims[0], plt_xlims[1], plt_ylims[0], plt_ylims[1]))
    if notgreat is not None:
        plt.scatter(notgreat[:, 0], notgreat[:, 1], edgecolors="r", marker='o', facecolors="none", linewidths=3,
                    s=markersize, zorder=2)
    if goodish is not None:
        plt.scatter(goodish[:, 0], goodish[:, 1], c='r', s=markersize, zorder=2)
    if best is not None:
        plt.scatter(best[:, 0], best[:, 1], marker="X", c="red", s=markersize, zorder=2)
    if not (env == constants.FETCH):
        plt.scatter(plt_gt[:, 0], plt_gt[:, 1], marker="*", c="lime", s=1500, zorder=1)

    plt.savefig(os.path.join(savedir, fname), bbox_inches="tight")
    plt.close("all")
