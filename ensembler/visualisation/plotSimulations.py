import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

import os, sys
sys.path.append(os.path.dirname(__file__)+"/..")

from ensembler.system import system
from ensembler.visualisation import style


def static_sim_plots(sys: system, x_range: tuple = None, title: str = "", out_path: str = None, resolution_full_space=style.potential_resolution) -> str:
    """
    Plot giving the sampled space, position distribution and forces
    :param sys:
    :param x_range:
    :param title:
    :param out_path:
    :return:
    """
    # gather data
    traj = sys.trajectory
    last_frame = traj.shape[0]-1

    x = list(traj.position)
    y = traj.totPotEnergy
    shift = traj.dhdpos 

    #dynamic plot range
    if (x_range == None):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(x_range) == range):
        x_pot = x_range
    else:
        x_pot = np.linspace(min(x_range), max(x_range)+1, resolution_full_space)

    ytot_space = sys.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)        #traj
    ax1.plot(x_pot, ytot_space, c=style.potential_light)
    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)   #start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)   #end_point

    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)   #start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)   #end_point
    print(viol)
    viol["bodies"][0].set_facecolor(color)

    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)

    #Labels
    ax1.set_ylabel("$V_pot$")
    ax1.set_xlabel("$x$")
    ax1.set_title("Potential Sampling")

    ax2.set_ylabel("$x$")
    ax2.set_xlabel("$simulation$")
    ax2.set_title("x-Distribution")

    ax3.set_ylabel("$dhdpos$")
    ax3.set_xlabel("$t$")
    ax3.set_title("Forces/shifts")

    ax2.set_xticks([])

    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    if(out_path):
        fig.savefig(out_path)
        plt.close(fig)

    return out_path, fig

def static_sim_plots_bias(sys: system, x_range: tuple = None,  y_range: tuple = None, title: str = "", out_path: str = None, resolution_full_space: int = style.potential_resolution) -> str:
    '''
    Plot giving the sampled space, position distribution and forces

    Parameters
    ----------
    sys: system Type
        The simulated system
    x_range: tuple
        Defines the range of the x axis of the first plot
    y_range: tuple
        Defines the range of the y axis of the first plot
    title: String
        Title for the plot
    out_path: str
        If specified, figure will be saved to this location
    resolution_full_space: int
        Number of points used for visualizing the potential space


    Returns
    -------
    out_path: str
        Location the figure is saved to
    fig: Figure object
    '''

    # gather data
    traj = sys.trajectory
    last_frame = traj.shape[0]-1

    x = list(traj.position)
    y = traj.totPotEnergy
    shift = traj.dhdpos

    #dynamic plot range
    if (x_range == None):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(x_range) == range):
        x_pot = x_range
    else:
        x_pot = np.linspace(min(x_range), max(x_range)+1, resolution_full_space)

    ytot_space = sys.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)        #traj

    #plot energy landscape of original and bias potential
    ytot_orig_space = sys.potential.origPotential.ene(x_pot)
    ytot_bias_space = sys.potential.addPotential.ene(x_pot)
    ax1.plot(x_pot, ytot_orig_space, c='red')
    ax1.plot(x_pot, ytot_bias_space, c=style.potential_light)

    #plot energy landscape of total potential
    ax1.plot(x_pot, ytot_space, c='blue')

    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)   #start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)   #end_point

    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)   #start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)   #end_point
    print(viol)
    viol["bodies"][0].set_facecolor(color)

    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)

    #Labels
    ax1.set_ylabel("$V_pot$")
    ax1.set_xlabel("$x$")
    ax1.set_title("Potential Sampling")

    #dynamic plot range
    if not (y_range == None):
        ax1.set_ylim(y_range)

    ax2.set_ylabel("$x$")
    ax2.set_xlabel("$simulation$")
    ax2.set_title("x-Distribution")

    ax3.set_ylabel("$dhdpos$")
    ax3.set_xlabel("$t$")
    ax3.set_title("Forces/shifts")

    ax2.set_xticks([])



    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    if(out_path):
        fig.savefig(out_path)
        plt.close(fig)

    return out_path, fig