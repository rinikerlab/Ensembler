import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

import os, sys
sys.path.append(os.path.dirname(__file__)+"/..")

from ensembler.system import system


def static_sim_plots(sys: system, x_range: tuple = None, title: str = "", out_path: str = None, resolution_full_space=1000) -> str:
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

    ax1.scatter(x, y, c="orange", alpha=0.8)        #traj
    ax1.scatter(x[0], y[0], c="green", alpha=0.8)   #start_point
    ax1.scatter(x[last_frame], y[last_frame], c="red", alpha=0.8)   #end_point
    ax1.plot(x_pot, ytot_space)

    ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c="green", alpha=0.8)   #start_point
    ax2.scatter([1], [x[last_frame]], c="red", alpha=0.8)   #end_point

    ax3.plot(range(len(x)), shift)

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