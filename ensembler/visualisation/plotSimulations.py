import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.colors import LogNorm

from ensembler.samplers import stochastic, newtonian, optimizers

from ensembler.potentials.OneD import metadynamicsPotential

from ensembler.util.ensemblerTypes import systemCls, Tuple, Union

from ensembler.visualisation import style
from ensembler.visualisation import plot_layout_settings

for key, value in plot_layout_settings.items():
    matplotlib.rcParams[key] = value


def simulation_analysis_plot(
    system: systemCls,
    title: str = "",
    out_path: str = None,
    limits_coordinate_space: Tuple[float, float] = None,
    oneD_limits_potential_system_energy: Tuple[float, float] = None,
    limits_force: Tuple[float, float] = None,
    twoD_number_of_bins: int = 25,
    resolution_full_space=style.potential_resolution,
    figsize: Tuple[float, float] = figaspect(0.25),
) -> Tuple[plt.figure, str]:
    """
    This is a wrapper function for the analysis of

    Parameters
    ----------
    system
    title
    out_path
    limits_coordinate_space
    resolution_full_space

    Returns
    -------

    """
    if system.nDimensions == 1:
        if hasattr(system, "bias_potential") and system.bias_potential:
            fig, out_path = oneD_biased_simulation_analysis_plot(
                system=system,
                title=title,
                out_path=out_path,
                limits_coordinate_space=limits_coordinate_space,
                limits_potential_system_energy=oneD_limits_potential_system_energy,
                limits_force=limits_force,
                resolution_full_space=resolution_full_space,
                figsize=figsize,
            )
        else:
            fig, out_path = oneD_simulation_analysis_plot(
                system=system,
                title=title,
                out_path=out_path,
                limits_coordinate_space=limits_coordinate_space,
                limits_potential_system_energy=oneD_limits_potential_system_energy,
                limits_force=limits_force,
                resolution_full_space=resolution_full_space,
                figsize=figsize,
            )
    elif system.nDimensions == 2:
        fig, out_path = twoD_simulation_analysis_plot(
            system=system,
            title=title,
            out_path=out_path,
            limits_coordinate_space=limits_coordinate_space,
            number_of_bins=twoD_number_of_bins,
            resolution_full_space=resolution_full_space,
            figsize=figsize,
        )

    return fig, out_path


def oneD_simulation_analysis_plot(
    system: systemCls,
    title: str = "",
    out_path: str = None,
    limits_coordinate_space: Tuple[float, float] = None,
    limits_potential_system_energy: Tuple[float, float] = None,
    limits_force: Tuple[float, float] = None,
    resolution_full_space=style.potential_resolution,
    figsize: Tuple[float, float] = figaspect(0.25),
) -> Tuple[plt.figure, str]:
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
    if isinstance(limits_coordinate_space, type(None)):
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif type(limits_coordinate_space) == range:
        x_pot = limits_coordinate_space
    else:
        x_pot = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_full_space)

    ytot_space = system.potential.ene(x_pot)

    # plot
    w, h = figsize
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    # plot coordinate exploration
    ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)  # traj
    ax1.plot(x_pot, ytot_space, c=style.potential_light)
    ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    if isinstance(system.potential, metadynamicsPotential):  # for metadynamics, show original potential
        ax1.plot(x_pot, system.potential.origPotential.ene(x_pot), c="k", alpha=style.alpha_val, zorder=10, label="original Potential")

    if not isinstance(limits_potential_system_energy, type(None)):
        ax1.set_ylim(limits_potential_system_energy)

    # plot position distribution
    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point
    viol["bodies"][0].set_facecolor(color)

    # plot force time series
    color = style.potential_color(3)
    ax3.plot(range(len(x)), shift, color=color)

    # visual_ranges:
    min_pos = min(x_pot) - min(x_pot) * 0.05
    max_pos = max(x_pot) + min(x_pot) * 0.05
    diff = max_pos - min_pos
    min_pos -= diff * 0.05
    max_pos += diff * 0.05

    ax1.set_xlim([min_pos, max_pos])
    ax2.set_ylim([min_pos, max_pos])

    if limits_force is not None:
        ax3.set_ylim([min(limits_force), max(limits_force)])

    # Labels
    ax1.set_ylabel("$V[kT]$")
    ax1.set_xlabel("$r$")
    ax1.set_title("Potential Sampling")

    ax2.set_ylabel("$r$")
    ax2.set_xlabel("$simulation$")
    ax2.set_title("Explored Space")

    ax3.set_xlabel("$t$")

    if issubclass(system.sampler.__class__, (stochastic.stochasticSampler, optimizers.optimizer)):
        ax3.set_title("Shifts")
        ax3.set_ylabel("$dr$")

    elif issubclass(system.sampler.__class__, (newtonian.newtonianSampler)):
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

    if out_path:
        fig.savefig(out_path)
        plt.close(fig)

    return fig, out_path


def oneD_biased_simulation_analysis_plot(
    system: systemCls,
    out_path: str = None,
    title: str = "",
    limits_coordinate_space: Tuple[float, float] = None,
    limits_potential_system_energy: Tuple[float, float] = None,
    limits_force: Tuple[float, float] = None,
    resolution_full_space: int = style.potential_resolution,
    figsize: Tuple[float, float] = figaspect(0.25),
) -> str:
    """
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
    """

    # gather data
    traj = system.trajectory
    last_frame = traj.shape[0] - 1

    x = list(traj.position)
    y = traj.total_potential_energy
    shift = traj.dhdpos

    # dynamic plot range
    if limits_coordinate_space is None:
        x_pot = np.linspace(min(x) + min(x) * 0.25, max(x) + max(x) * 0.25, resolution_full_space)
    elif type(limits_coordinate_space) == range:
        x_pot = limits_coordinate_space
    else:
        x_pot = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_full_space)

    ytot_space = system.potential.ene(x_pot)

    # plot
    w, h = figsize
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[w, h])

    # traj

    # plot energy landscape of original and bias potential
    if isinstance(system.potential, metadynamicsPotential):
        # special figure for metadynamics simulations
        # plot energy landscape of original potential
        ytot_orig_space = system.potential.origPotential.ene(x_pot)
        ax1.plot(x_pot, ytot_orig_space, c="red")
        # plot energy landscape of total potential
        # ytot_bias_space = sys.potential.origPotential.ene(x_pot) + sys.potential.finished_steps* sys.potential.addPotential.ene(x_pot)
        # ax1.plot(x_pot, ytot_bias_space, c='blue')

        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)
        ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
        ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point
    else:
        ax1.scatter(x, y, c=style.trajectory_color, alpha=style.alpha_traj)

        oP = system.potential.origPotential
        oP._update_functions()
        ytot_orig_space = oP.ene(x_pot)
        ytot_bias_space = system.potential.addPotential.ene(x_pot)

        ax1.plot(x_pot, ytot_orig_space, c="red")
        ax1.plot(x_pot, ytot_bias_space, c=style.potential_light)

        # plot energy landscape of total potential
        ax1.plot(x_pot, ytot_space, c="blue")

        ax1.scatter(x[0], y[0], c=style.traj_start, alpha=style.alpha_val)  # start_point
        ax1.scatter(x[last_frame], y[last_frame], c=style.traj_end, alpha=style.alpha_val)  # end_point

    color = style.potential_color(2)
    viol = ax2.violinplot(x, showmeans=False, showextrema=False)
    ax2.boxplot(x)
    ax2.scatter([1], [x[0]], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax2.scatter([1], [x[last_frame]], c=style.traj_end, alpha=style.alpha_val)  # end_point

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

    if limits_force is not None:
        ax3.set_ylim([min(limits_force), max(limits_force)])

    # labels
    ax2.set_ylabel("$x$")
    ax2.set_xlabel("$simulation$")
    ax2.set_title("x-Distribution")

    ax3.set_ylabel("$dhdpos$")
    ax3.set_xlabel("$t$")
    ax3.set_title("Forces/shifts")

    ax2.set_xticks([])

    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path)
        plt.close(fig)

    return out_path, fig


def twoD_simulation_analysis_plot(
    system: systemCls,
    out_path: str = None,
    title: str = "",
    limits_coordinate_space: Tuple[Tuple[float, float], Tuple[float, float]] = [[-180, 180], [-180, 180]],
    number_of_bins: int = 25,
    limits_force: Tuple[float, float] = None,
    resolution_full_space: int = style.potential_resolution,
    figsize: Tuple[float, float] = figaspect(0.25),
) -> Tuple[plt.Figure, Union[str, None]]:
    """
         Plot giving the sampled space, position histogram and forces of a 2D simulation

     Parameters
     ----------
     system: sustemCls
             The simulated 2D system
     out_path: str, optional
         save the plot to path
     title:str, optional
         title of the plot
     limits_coordinate_space: Tuple[Tuple[float, float], Tuple[float, float]], optional
         range of the coordinate space: ((xmin, xmax), (ymin, ymax))
     number_of_bins:
         number of bins for the position histogram
     resolution_full_space: int
         Number of points used for visualizing the potential space
     Returns
     -------


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
     plt.Figure
         Figure object
     Union[str, None]
         out_path
    """

    # get system information
    traj_pos = np.array(list(map(lambda x: np.array(x), system.trajectory.position))).T
    force1, force2 = np.array(list(map(lambda x: np.array(x), system.trajectory.dhdpos))).T
    potential = system.potential

    # dynamic plot range
    if limits_coordinate_space is None:
        limits_coordinate_space = np.array([(min(traj_pos[0]), max(traj_pos[0])), (min(traj_pos[1]), max(traj_pos[1]))])
    else:
        limits_coordinate_space = np.array(limits_coordinate_space)

    positionsX = np.linspace(limits_coordinate_space[0][0], limits_coordinate_space[0][1], resolution_full_space)
    positionsY = np.linspace(limits_coordinate_space[1][0], limits_coordinate_space[1][1], resolution_full_space)
    x_positions, y_positions = np.meshgrid(positionsX, positionsY)
    positions2D = np.array([x_positions.flatten(), y_positions.flatten()]).T

    # plot
    w, h = figsize
    fig = plt.figure(figsize=[w, h])
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])

    # traj
    # plot energy landscape of total potential
    ene_space = potential.ene(positions2D).reshape([resolution_full_space, resolution_full_space])
    ax1.imshow(ene_space[::-1], extent=[*limits_coordinate_space.flat])
    ax1.scatter(*traj_pos, c=style.trajectory_color, alpha=style.alpha_traj)
    ax1.scatter(*traj_pos[:, 0], c=style.traj_start, alpha=style.alpha_val)  # start_point
    ax1.scatter(*traj_pos[:, -1], c=style.traj_end, alpha=style.alpha_val)  # end_point

    if limits_coordinate_space is not None:
        ax1.set_ylim(limits_coordinate_space[0])
        ax1.set_ylim(limits_coordinate_space[1])

    # position density:
    x, y = traj_pos
    hist = ax2.hist2d(*traj_pos, bins=number_of_bins, range=limits_coordinate_space, density=True, norm=LogNorm())
    cb = plt.colorbar(hist[3], ax=ax2)

    # forces:
    ax3.plot(range(len(force1)), force1)
    ax4.plot(range(len(force2)), force2)

    # labels&titles
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$x$")
    ax1.set_title("Potential Sampling")

    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_title("Sampling Histogramm")

    if issubclass(system.sampler.__class__, newtonian.newtonianSampler) or issubclass(system.sampler.__class__, optimizers.optimizer):
        ax3.set_title("Forces Timeseries")
        ax3.set_ylabel("$dhdpos_1$")
        ax4.set_ylabel("$dhdpos_2$")

    if issubclass(system.sampler.__class__, stochastic.stochasticSampler):
        ax3.set_title("Shift Timeseries")
        ax3.set_ylabel("$dr_1$")
        ax4.set_ylabel("$dr_2$")

    ax4.set_xlabel("$t$")

    cb.set_label("$p$")

    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig, out_path
