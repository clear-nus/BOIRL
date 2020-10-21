import numpy as np
import matplotlib.pyplot as plt
import os.path
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib

def int_to_point(i, grid_size):
    """
    Convert a state int into the corresponding coordinate.
    i: State int.
    -> (x, y) int tuple.
    """
    return (i % grid_size, i // grid_size)


def point_to_int(self, p):
    """
    Convert a coordinate into the corresponding state int.
    p: (x, y) tuple.
    -> State int.
    """
    return p[0] + p[1] * self.grid_size

def show_values(pc, roundval = True, fmt="%f", **kw):

    pc.update_scalarmappable()
    #ax = pc.get_axes()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        """
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        """
        if value > 30.:
            color = (0.0,0.0,0.0)
        else:
            color = (1.0, 1.0, 1.0)

        font = {'family': 'serif',
                'color': color,
                'weight': 'bold',
                'size': 10,
                'ha' : 'left',
                'va' : 'top'
                }
        if roundval:
            ax.text(x-0.5, y+0.5, int(value), fontdict = font)
        else:
            ax.text(x - 0.5, y + 0.5, round(value, 2), fontdict = font)


def plot_rewards(plot_number,plot_location,plot_title,reward,grid_size,current_goal):
    if not os.path.exists(plot_location):
        os.makedirs(plot_location)
    plt.figure(plot_number, figsize=(18, 10), dpi=80)
    p1 = plt.pcolormesh(reward.reshape((grid_size, grid_size)), cmap="hot", vmin=np.min(reward), vmax=np.max(reward), edgecolor="white")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Temperature in $^\circ$C")
    plt.title(plot_title, fontsize="26")  # ,color='#D63569')
    # plt.suptitle(" Environment:" + str(eind))

    font = {'family': 'serif',
            'color': 'white',
            'weight': 'extra bold',
            'size': 10,
            'verticalalignment': 'center',
            'horizontalalignment': 'center'
            }
    show_values(p1)
    plt.text(0.5, 0.5, "S", fontdict=font)
    gx, gy = int_to_point(current_goal, grid_size)
    plt.text(gx + 0.5, gy + 0.5, "G", fontdict=font)
    plt.savefig(os.path.join(plot_location, str(plot_number) + ".png"), bbox_inches='tight')

def draw_traj(pc, env, trajectories, goals, grid_size=5, arrow_colors=None, threshold_plt = False, plot_gradient = False, temp_threshold = None,width = 0.025,legend_plot=False,discontinous_agents = False, agent_names = None,start_state=0):
    head_length = 0.12
    width = width
    head_width = width/0.3

    #pc.update_scalarmappable()
    #ax = pc.get_axes()
    ax = pc.axes
    def int_to_point(i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """
        return (i % grid_size, i // grid_size)

    sh = np.shape(trajectories)
    if len(sh) == 2:
        trajectories = np.expand_dims(trajectories,axis=0)
    #temperatures = np.array(env[:,1])
    temperatures = np.array(env[:, 2])
    temperatures2 = np.array(temperatures)
    if threshold_plt:
        temperatures2[np.where(temperatures <= temp_threshold)] = 0
        if not plot_gradient:
            temperatures2[np.where(temperatures > temp_threshold)] = 1

    cmap = matplotlib.colors.ListedColormap(['#FF4136', '#0074D9', '#2ECC40'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    #p0 = ax.pcolormesh(temperatures.reshape((5,5)),cmap="hot",vmin=0,vmax=90, edgecolor="white")
    #ax.pcolormesh(temperatures2.reshape((5,5)),cmap="hot",vmin=0,vmax=90, edgecolor="white")
    p0 = ax.pcolormesh(temperatures.reshape((grid_size, grid_size)), cmap=cmap,norm=norm, vmin=-1.5, vmax=1.5, edgecolor="white")
    ax.pcolormesh(temperatures2.reshape((grid_size, grid_size)), cmap=cmap, norm=norm,vmin=-1.5, vmax=1.5, edgecolor="white")

    cmap = plt.cm.hot
    cNorm = colors.Normalize(vmin=0, vmax=90)  #
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    """
    arr_cmap = plt.cm.tab20
    arr_cNorm = colors.Normalize(vmin=0, vmax=5)  #
    arr_scalarMap = cmx.ScalarMappable(norm=arr_cNorm, cmap=arr_cmap)
    """

    for tauind, tau in enumerate(trajectories):
        if (arrow_colors==None):
            arr_color = "red"
        else:
            arr_color = arrow_colors[tauind]
            #arr_color = arr_scalarMap.to_rgba(tauind)
        for sind in range(np.shape(tau)[0]-1):
            s0 = tau[sind,0]
            s1 = tau[sind+1,0]
            x0,y0 = int_to_point(s0)
            x1,y1 = int_to_point(s1)
            arrow_start_x = x0 + 0.5+(tauind + 1)/50.
            arrow_start_y = y0 + 0.5+(tauind + 1)/50.
            arrow_dx = (x1-x0)#*(1 - head_length)
            arrow_dy = (y1-y0)#*(1 - head_length)
            if not(arrow_dx == 0 and arrow_dy == 0):
                ax.arrow(arrow_start_x,arrow_start_y,arrow_dx,arrow_dy,color = arr_color,width=width, head_width=head_width,head_length=head_length,length_includes_head=True)
    scalarMap.set_array(temperatures)

    if not threshold_plt:
        cb = plt.colorbar(scalarMap)
        cb.ax.tick_params(labelsize=16)
        cb.set_label("Temperature in $^\circ$C")

    start_x, start_y = int_to_point(start_state)
    #goal = np.where(env[:, 2] == 1)[0][0]
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'extra bold',
            'size': 10,
            'verticalalignment': 'center',
            'horizontalalignment': 'center'
            }
    ax.text(start_x + 0.5, start_y + 0.5, "S", ha="center", va="center", fontdict = font)
    for goal in goals:
        goal_x, goal_y = int_to_point(goal)
        ax.text(goal_x+0.5, goal_y+0.5, "G", ha="center", va="center", fontdict = font)
    show_values(p0)
    if legend_plot:
        all_patches =[]
        for cind,c in enumerate(arrow_colors):
            if cind < len(temp_threshold):
                ag_name = str(agent_names[cind]) if discontinous_agents else str(cind)
                all_patches.append(mpatches.Patch(color=c, label= "Agent " + ag_name +" w Thresh: "+str(temp_threshold[cind])))
        plt.legend(handles=all_patches,shadow=True, fancybox=True,title="Arrow Legend")

plt.close("all")