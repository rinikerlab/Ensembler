import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect


from ensembler.samplers import stochastic, newtonian, optimizers

from ensembler.visualisation import style
from ensembler.potentials.biased_potentials.biasOneD import metadynamicsPotential

from ensembler.util.ensemblerTypes import systemCls, Tuple


def oneD_simulation_analysis_plot(system: systemCls, title: str = "", out_path: str = None,
                                  limits_coordinate_space: Tuple[float, float] = None, limits_potential_system_energy: Tuple[float, float] = None, limits_force: Tuple[float, float] = None,
                                  resolution_full_space=style.potential_resolution) -> Tuple[plt.figure, str]:
    """
        This plot gives insight into sampled coordinate - space, the position distribution and the force timeseries

    Parameters
    ----------
    system: systemCls
        a system that carried out a simulation, which should be visualized.
    title: str, optional
        title to the output plot
    out_path: str, optional
        store the plot to the given path
    limits_coordinate_space: tuple, optional
        the coordinate space range
    limits_potential_system_energy: tuple, optional
        y-limits for the Potential energies
    limits_force: tuple, optional
        y-limits for force plots
    resolution_full_space: int, optional
        how many points, shall be used to visualize the potential function.

    Returns
    -------
    Tuple[plt.figure, str]
        returns the generated figure and the output str
    """

    # gather data
    traj = system.trajectory
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.total_potential_energy
    shift = traj.dhdpos

    # dynamic plot range
    if (isinstance(limits_coordinate_space, type(None))):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(limits_coordinate_space) == range):
        x_pot = limits_coordinate_space
    else:
        x_pot = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_full_space)

    ytot_space = system.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    #plot coordinate exploration
    ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)  # traj
    ax1.plot(x_pot, ytot_space, c=style.potential_light)
    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    if(isinstance(system.potential, metadynamicsPotential)):   #for metadynamics, show original potential
        ax1.plot(x_pot, system.potential.origPotential.ene(x_pot), c="k", alpha=style.alpha_val, zorder=10, label="original Potential")

    if (not isinstance(limits_potential_system_energy, type(None))):
        print(limits_potential_system_energy)
        ax1.set_ylim(limits_potential_system_energy)

    # plot position distribution
    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point
    viol["bodies"][0].set_facecolor(color)

    #plot force time series
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

    if(limits_force is not None):
        ax3.set_ylim([min(limits_force), max(limits_force)])
        

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


def oneD_biased_simulation_analysis_plot(system: systemCls, out_path: str = None, title: str = "",
                                         limits_coordinate_space: Tuple[float, float] = None, limits_potential_system_energy:Tuple[float, float] = None, limits_force: Tuple[float, float] = None,
                                         resolution_full_space: int = style.potential_resolution) -> str:
    '''
    Plot giving the sampled space, position distribution and forces

    Parameters
    ----------
    sys: system Type
        The simulated system
    limits_potential_system_energy: tuple
        Defines the range of the x axis of the first plot
    limits_potential_system_energy: tuple
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
    traj = system.trajectory
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.total_potential_energy
    shift = traj.dhdpos

    # dynamic plot range
    if (limits_coordinate_space is None):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(limits_coordinate_space) == range):
        x_pot = limits_coordinate_space
    else:
        x_pot = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_full_space)

    ytot_space = system.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    # traj

    # plot energy landscape of original and bias potential
    if isinstance(system.potential, metadynamicsPotential):
        # special figure for metadynamics simulations
        # plot energy landscape of original potential
        ytot_orig_space = system.potential.origPotential.ene(x_pot)
        ax1.plot(x_pot, ytot_orig_space, c='red')
        # plot energy landscape of total potential
        # ytot_bias_space = sys.potential.origPotential.ene(x_pot) + sys.potential.finished_steps* sys.potential.addPotential.ene(x_pot)
        # ax1.plot(x_pot, ytot_bias_space, c='blue')

        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)
        ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
        ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point
    else:
        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)
        print(system)
        print(system.potential.origPotential)
        ytot_orig_space = system.potential.origPotential.ene(x_pot)
        ytot_bias_space = system.potential.addPotential.ene(x_pot)
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
    if not (limits_potential_system_energy is None):
        ax1.set_ylim(limits_potential_system_energy)

    if (limits_force is not None):
        ax3.set_ylim([min(limits_force), max(limits_force)])

    #labels
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

def Two_biased_simulation_analysis_plot(system: systemCls, out_path: str = None, title: str = "",
                                         limits_coordinate_space: Tuple[float, float] = None, limits_potential_system_energy:Tuple[float, float] = None, limits_force: Tuple[float, float] = None,
                                         resolution_full_space: int = style.potential_resolution) -> str:
    '''
    Plot giving the sampled space, position distribution and forces

    Parameters
    ----------
    sys: system Type
        The simulated system
    limits_potential_system_energy: tuple
        Defines the range of the x axis of the first plot
    limits_potential_system_energy: tuple
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


    test_timing_with_points=100

    positions = np.linspace(-180, 180, test_timing_with_points)
    x_positions, y_positions = np.meshgrid(positions,positions)
    positions2D = np.array([x_positions.flatten(), y_positions.flatten()]).T

    traj_pos = np.array(list(map(lambda x: np.array(x), sys.trajectory.position))).T
    potential = system.potential


    # gather data
    traj = system.trajectory
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.total_potential_energy
    shift = traj.dhdpos

    # dynamic plot range
    if (limits_coordinate_space is None):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif (type(limits_coordinate_space) == range):
        x_pot = limits_coordinate_space
    else:
        x_pot = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_full_space)

    ytot_space = system.potential.ene(x_pot)

    # plot
    w, h = figaspect(0.25)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    # traj
    # plot energy landscape of total potential
    ax1.imshow(potential.ene(positions2D).reshape([test_timing_with_points,test_timing_with_points]), extent=[*space_range, *space_range])

    ax1.scatter(*traj_pos, c=style.trajectory_color, alpha=style.alpha_traj)
    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    ax1.set_ylabel("$V_pot$")
    ax1.set_xlabel("$x$")
    ax1.set_title("Potential Sampling")

    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    if (out_path):
        fig.savefig(out_path)
        plt.close(fig)
    return out_path, fig
    """
    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point
    print(viol)
    viol["bodies"][0].set_facecolor(color)

    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)


    # dynamic plot range
    if not (limits_potential_system_energy is None):
        ax1.set_ylim(limits_potential_system_energy)

    if (limits_force is not None):
        ax3.set_ylim([min(limits_force), max(limits_force)])

    # Labels
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


"""

