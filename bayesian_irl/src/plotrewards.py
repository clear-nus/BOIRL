import numpy as np
import matplotlib.pyplot as plt
import os
#import seaborn as sns
from tqdm import tqdm
#sns.set(font_scale = 3)
env = "Borlange"
loaddir = os.path.join("Results", env)
ntrials = 10
savedir = "PlotRewards"
os.makedirs(savedir, exist_ok=True)
fontlabs = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 50,
            'verticalalignment': 'center',
            'horizontalalignment': 'center'
            }
for tr in tqdm(range(ntrials)):
    r = os.path.join(loaddir, "rewards%d.npy") % tr
    rewards = np.load(r)

    plt.figure(tr, figsize=(15.25, 13.5), dpi=80)
    sns_plot = sns.kdeplot(rewards[:, 0], rewards[:, 1], shade=True, cmap="viridis_r",shade_lowest=False)
    fig = sns_plot.get_figure()



    """
    sns_plot.set_xticks(
        (np.amin(rewards[:, 0]), np.amax(rewards[:, 0]), (np.amin(rewards[:, 0]) + np.amax(rewards[:, 0])) / 2),
        size=30)
    sns_plot.set_yticks(
        (np.amin(rewards[:, 1]), np.amax(rewards[:, 1])(np.amin(rewards[:, 1]) + np.amax(rewards[:, 1])) / 2),
        size=30)
    """
    """
    plt.xticks(
        (np.amin(rewards[:, 0]), np.amax(rewards[:, 0]), (np.amin(rewards[:, 0]) + np.amax(rewards[:, 0])) / 2),
        fontsize=30)
    plt.yticks(
        (np.amin(rewards[:, 1]), np.amax(rewards[:, 1]), (np.amin(rewards[:, 1]) + np.amax(rewards[:, 1])) / 2),
        fontsize=30)
    """
    if env == "Borlange":
        plt.xlim((-2.5, 0))
        plt.ylim((-2.5, 0))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        #plt.xticks((-2.5, -1.2, 0.))
        #plt.yticks((-2.5, -1.2, 0.))
        sns_plot.set_xlabel("Time, " + r"$\theta_0$", fontdict=fontlabs, labelpad=30)
        sns_plot.set_ylabel("Left-Turn, " + r"$\theta_1$", fontdict=fontlabs, labelpad=15)
        plt.scatter(x=[-2], y=[-1], marker="*", color="lime",s=1500)
        #sns_splot.set_xticks([-2.5, -1.2, 0.])
        #sns_splot.set_yticks([-2.5, -1.2, 0.])
        #plt.xticks(fontsize=30)
        #plt.yticks(fontsize=30)
        plt.annotate(r"$\theta_2=-1$", (-2.4, -2.4), fontsize=40, c="black")

    elif env == "Gridworld":
        sns_plot.set_xlabel("Steepness, " + r"$\theta_0$", fontdict=fontlabs, labelpad=30)
        sns_plot.set_ylabel("Midpoint, " + r"$\theta_1$", fontdict=fontlabs, labelpad=15)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim((-2, 2))
        plt.ylim((-10, 10))
        #sns_plot.set_xticks((-2, 0., 2.))
        #sns_plot.set_yticks((-10., 0., 10.))
        plt.annotate(r"$\theta_2=0$", (-1.8, -9), fontsize=40, c="black")
        plt.scatter([1.25],[5],marker="*",c="lime",s=1500)


    #plt.colorbar(sns_plot)
    fig.savefig(os.path.join(loaddir, "rewards%d.png") % tr,bbox_inches="tight")
    plt.close("all")



"""
for s in range(rewards.shape[1]):
    fig = plt.figure(s,figsize=(24,13.5),dpi=80)
    plt.hist(rewards[:, s], range=(-10, 10))
    plt.savefig(os.path.join(savedir,"reward_state%d")%s,bbox_inches="tight")
    if s%10 == 0:
        plt.close("all")
plt.close("all")
"""
