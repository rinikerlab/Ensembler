import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

sys.path.append(os.path.dirname(__file__) + "/..")

from ensembler.samplers import stochastic, newtonian, optimizers

from ensembler.system import system
from ensembler.visualisation import style
from ensembler.potentials.biased_potentials.biasOneD import metadynamicsPotential


def static_sim_plots(sys: system, x_range: tuple = None, y_lim_Pot: tuple = None, title: str = "", out_path: str = None,
                     resolution_full_space=style.potential_resolution) -> str:
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
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.totPotEnergy
    shift = traj.dhdpos

    # dynamic plot range
    if (isinstance(x_range, type(None))):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(x_range) == range):
        x_pot = x_range
    else:
        x_pot = np.linspace(min(x_range), max(x_range) + 1, resolution_full_space)

    ytot_space = sys.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)  # traj
    ax1.plot(x_pot, ytot_space, c=style.potential_light)
    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    if(isinstance(sys.potential, metadynamicsPotential)):   #for metadynamics, show original potential
        ax1.plot(x_pot, sys.potential.origPotential.ene(x_pot), c="k", alpha=style.alpha_val, zorder=10, label="original Potential")

    if (not isinstance(y_lim_Pot, type(None))):
        ax1.set_ylim(y_lim_Pot)

    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point
    viol["bodies"][0].set_facecolor(color)

    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)

    #visual_ranges:
    min_pos = min(x_pot)-min(x_pot)*0.05
    max_pos = max(x_pot)+min(x_pot)*0.05
    diff = max_pos-min_pos
    min_pos -= diff * 0.05
    max_pos += diff * 0.05
    ax1.set_xlim([min_pos, max_pos])
    ax2.set_ylim([min_pos, max_pos])

    # Labels
    ax1.set_ylabel("$V[kT]$")
    ax1.set_xlabel("$r$")
    ax1.set_title("Potential Sampling")

    ax2.set_ylabel("$r$")
    ax2.set_xlabel("$simulation$")
    ax2.set_title("Explored Space")

    ax3.set_xlabel("$t$")

    if (issubclass(system.sampler.__class__, (stochastic.stochasticSampler, optimizers.optimizer))):
        ax3.set_title("Shifts")
        ax3.set_ylabel("$dr$")

    elif (issubclass(system.sampler.__class__, (newtonian.newtonianSampler))):
        ax3.set_title("Forces")
        ax3.set_ylabel("$\partial V/ \partial r$")

    else:
        ax3.set_title("Shifts")  # FIX this part!
        ax3.set_ylabel("$dr$")
        # raise Exception("Did not find samplers type  >"+str(system.samplers.__class__)+"< ")

    ax2.set_xticks([])


    fig.suptitle(title, y=1.1)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()

    if (out_path):
        fig.savefig(out_path)
        plt.close(fig)

    return fig, out_path


def static_sim_plots_bias(sys: system, x_range: tuple = None, y_range: tuple = None, title: str = "",
                          out_path: str = None, resolution_full_space: int = style.potential_resolution) -> str:
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
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.totPotEnergy
    shift = traj.dhdpos

    # dynamic plot range
    if (x_range is None):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(x_range) == range):
        x_pot = x_range
    else:
        x_pot = np.linspace(min(x_range), max(x_range) + 1, resolution_full_space)

    ytot_space = sys.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    # traj

    # plot energy landscape of original and bias potential
    if isinstance(sys.potential, metadynamicsPotential):
        # special figure for metadynamics simulations
        # plot energy landscape of original potential
        ytot_orig_space = sys.potential.origPotential.ene(x_pot)
        ax1.plot(x_pot, ytot_orig_space, c='red')
        # plot energy landscape of total potential
        # ytot_bias_space = sys.potential.origPotential.ene(x_pot) + sys.potential.finished_steps* sys.potential.addPotential.ene(x_pot)
        # ax1.plot(x_pot, ytot_bias_space, c='blue')

        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)
        ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
        ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point
    else:
        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)

        ytot_orig_space = sys.potential.origPotential.ene(x_pot)
        ytot_bias_space = sys.potential.addPotential.ene(x_pot)
        ax1.plot(x_pot, ytot_orig_space, c='red')
        ax1.plot(x_pot, ytot_bias_space, c=style.potential_light)

        # plot energy landscape of total potential
        ax1.plot(x_pot, ytot_space, c='blue')

        ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
        ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point
    print(viol)
    viol["bodies"][0].set_facecolor(color)

    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)

    # Labels
    ax1.set_ylabel("$V_pot$")
    ax1.set_xlabel("$x$")
    ax1.set_title("Potential Sampling")

    # dynamic plot range
    if not (y_range is None):
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

    if (out_path):
        fig.savefig(out_path)
        plt.close(fig)

    return out_path, fig
