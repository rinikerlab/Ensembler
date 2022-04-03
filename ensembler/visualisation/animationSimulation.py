import numpy as np
from matplotlib import animation, pyplot as plt

from ensembler.util.ensemblerTypes import systemCls, Iterable, List, Tuple, Union, Number
from ensembler.visualisation import style, dpi_animation, animation_figsize
from ensembler.potentials.OneD import metadynamicsPotential


def animation_trajectory(
    simulated_system: systemCls,
    limits_coordinate_space: Tuple[float, float] = None,
    limits_potential_system_energy: Tuple[float, float] = None,
    resolution_of_analytic_potential: int = 1000,
    title: str = None,
    out_path: str = None,
    out_writer: str = "pillow",
    dpi: int = dpi_animation,
    every_n_frame: int = 1,
) -> Tuple[animation.Animation, Union[str, None]]:
    """
        this function is generating a animation out of a simulation.
    Parameters
    ----------
    simulated_system: systemCls
        the system, that carried out a simulation to be visualized
    limits_coordinate_space: Tuple[float, float], optional
        the plot limits of the coordinate space
    limits_potential_system_energy: Tuple[float, float], optional
        the plot limits of the potential energies
    title: str, optional
        the plot title
    out_path: str, optional
        the out  path storing the animation.
    out_writer: str, optional
        the animation render.
    dpi: int, optional
        the resolution of the animation.
    every_n_frame: int, optional
        use every n-th frame

    Returns
    -------
    Tuple[animation.Animation, Union[str, None]]
        returns the animation object and the output path if saved.
    """

    # plotting
    x1data = simulated_system.trajectory.position
    y1data = simulated_system.trajectory.total_potential_energy
    shift = simulated_system.trajectory.dhdpos

    lastindx = x1data.shape[-1] - 1
    lastindy = np.array(list(y1data)).shape[-1] - 1

    x_max = max(x1data)
    x_min = min(x1data)

    if limits_coordinate_space is None:
        xtot_space = np.linspace(x_min + 0.2 * x_min, x_max + 0.2 * x_max + 1, resolution_of_analytic_potential)
    else:
        xtot_space = np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, resolution_of_analytic_potential)
    ytot_space = simulated_system.potential.ene(xtot_space)

    # settings
    step_size = every_n_frame
    tmax = len(y1data) - 1
    t0 = 0
    active_dots = 20

    # build figure
    fig = plt.figure(dpi=60, figsize=animation_figsize)
    ax = fig.add_subplot(111)

    ## setup static parts
    ax.plot(xtot_space, ytot_space, label="Potential", c=style.potential_light)

    if isinstance(simulated_system.potential, metadynamicsPotential):  # for metadynamics, show original potential
        ax.plot(
            xtot_space,
            simulated_system.potential.origPotential.ene(xtot_space),
            c="k",
            alpha=style.alpha_val,
            zorder=10,
            label="original Potential",
        )

    ### Params
    if limits_coordinate_space != None:
        ax.set_xlim(limits_coordinate_space)
    if limits_potential_system_energy != None:
        ax.set_ylim(limits_potential_system_energy)

    ax.set_xlabel("$r$")
    ax.set_ylabel("$V$")

    if title != None:
        fig.suptitle(title)

    # data structures in ani
    ##setup data structures
    xdata, ydata = [], []
    scatter = ax.scatter([], [], c=[], vmin=0, vmax=1, cmap=style.animation_traj)

    (start_p,) = ax.plot([], [], "bo", c=style.traj_start, ms=10)
    (end_p,) = ax.plot([], [], "bo", c=style.traj_end, ms=10)
    (curr_p,) = ax.plot([], [], "bo", c=style.traj_current, ms=10)

    def init():
        del xdata[:], ydata[:]
        start_p.set_data(x1data[0], y1data[0])
        end_p.set_data([], [])
        curr_p.set_data([], [])

    def data_gen(t=t0):
        while t < tmax:
            t += step_size
            yield x1data[t], y1data[t]

    def run(data):
        x, V = data

        if x == x1data[lastindx]:
            curr_p.set_data([], [])
            end_p.set_data(x1data[lastindx], y1data[lastindy])
        else:
            curr_p.set_data([x], [V])
            xdata.append(x)
            ydata.append(V)

            # color fading effect
            if len(xdata) > active_dots:
                c = np.concatenate((np.array([0.6 for x in range(len(xdata) - active_dots)]), np.linspace(0.6, 0, active_dots)))
            else:
                c = np.linspace(0.6, 0, len(xdata))

            # set new data
            scatter.set_offsets(np.c_[xdata, ydata])
            scatter.set_array(c)

        # if necessary adapt y axis
        if min(ax.get_ylim()) > V:
            ax.set_ylim(V + V * 0.1, max(ax.get_ylim()))

        return (scatter,)

    ani = animation.FuncAnimation(
        fig=fig, func=run, frames=data_gen, init_func=init, blit=False, interval=20, repeat=False, cache_frame_data=True
    )
    if out_path != None:
        # Set up formatting for the movie files
        Writer = animation.writers[out_writer]
        writer = Writer(metadata=dict(artist="animationsMD1D_David_Hahn_Benjamin_Schroeder"))
        ani.save(out_path, writer=writer, dpi=dpi)

    return ani, out_path


def animation_EDS_trajectory(
    system: systemCls,
    limits_coordinate_space=None,
    limits_potential_system_energy: Tuple[float, float] = None,
    title: str = None,
    out_path: str = None,
    hide_legend: bool = True,
    s_values: List[float] = [1.0],
    step_size: float = 1,
    out_writer: str = "pillow",
    dpi: int = 100,
    total_potential_resolution_points: int = 100,
) -> (animation.Animation, (str or None)):
    """
        this function is generating a animation out of an eds-simulation.

    Parameters
    ----------
    system: systemCls
        the system, that carried out a simulation to be visualized
    limits_coordinate_space: Tuple[float, float], optional
        the plot limits of the coordinate space
    limits_potential_system_energy: Tuple[float, float], optional
        the plot limits of the potential energies
    title: str, optional
        the plot title
    out_path: str, optional
        the out  path storing the animation.
    hide_legend: bool, optional
        should the plot legend be hidden?
    s_values: List[float], optional
        s-values for the eds-potential
    step_size: int, optional
        size of the steps trough the trajectory
    out_writer: str, optional
        the animation render.
    dpi: int, optional
        the resolution of the animation.
    total_potential_resolution_points: int, optional
        the number of points for the visualization of the analytical reference potential.

    Returns
    -------
    Tuple[animation.Animation, Union[str, None]]
        returns the animation object and the output path if saved.
    """

    # plotting
    x1data = np.array(system.trajectory.position)
    y1data = system.trajectory.totPotEnergy

    x_max = max(x1data)
    x_min = min(x1data)
    active_dots = 20

    if limits_coordinate_space is None:
        xtot_space = np.array(np.arange(x_min + 0.2 * x_min, x_max + 0.2 * x_max + 1), ndmin=1)
    else:
        xtot_space = np.array(
            np.linspace(min(limits_coordinate_space), max(limits_coordinate_space) + 1, total_potential_resolution_points), ndmin=1
        )

    tmax = len(y1data) - 1 - step_size
    t0 = 0

    xdata, ydata = [], []

    # figures
    fig = plt.figure(figsize=animation_figsize)
    ax = fig.add_subplot(111)

    from ensembler.visualisation.plotPotentials import envPot_differentS_overlay_plot

    _, ax = envPot_differentS_overlay_plot(
        eds_potential=system.potential, s_values=s_values, title=title, positions=xtot_space, axes=ax, hide_legend=hide_legend
    )

    scatter = ax.scatter([], [], c=[], vmin=0, vmax=1, cmap="inferno")
    (start_p,) = ax.plot([], [], "bo", c="g", ms=10)
    (end_p,) = ax.plot([], [], "bo", c="r", ms=10)
    (curr_p,) = ax.plot([], [], "bo", c="k", ms=10)

    # Params
    ax.set_xlabel("$r$")
    ax.set_ylabel("$V$")
    if isinstance(title, type(None)):
        fig.suptitle(title)

    def init():
        del xdata[:], ydata[:]

        end_p.set_data([], [])
        curr_p.set_data([], [])

        if limits_coordinate_space != None:
            ax.set_xlim(limits_coordinate_space)
        # return line,

    def data_gen(t=t0):
        while t < tmax:
            t += step_size
            yield x1data[t], y1data[t]

    def run(data):
        # update the data
        x, V = data
        if type(x) == type(x1data[-1]) == list and all([xi == x1i for xi, x1i in zip(x, x1data[-1])]):  # last step of traj
            curr_p.set_data([], [])
            end_p.set_data(x1data[-1], y1data[-1])
        elif type(x) == type(x1data[-1]) == Number and x == x1data[-1]:  # last step of traj
            curr_p.set_data([], [])
            end_p.set_data(x1data[-1], y1data[-1])
        else:
            curr_p.set_data([x], [V])
            xdata.append(x) if (not isinstance(x, Iterable)) else xdata.append(x[0])
            ydata.append(V)

            if len(xdata) > active_dots + 10:
                c = np.concatenate((np.array([0.6 for x in range(len(xdata) - active_dots)]), np.linspace(0.6, 0, active_dots)))
            else:
                c = np.linspace(0.6, 0, len(xdata))

            scatter.set_offsets(np.c_[xdata, ydata])
            scatter.set_array(c)

        if min(ax.get_ylim()) > V:
            ax.set_ylim(V + V * 0.1, max(ax.get_ylim()))

    ani = animation.FuncAnimation(
        fig=fig, func=run, frames=data_gen, init_func=init, blit=False, interval=20, repeat=False, save_count=len(x1data)
    )
    if out_path != None:
        # Set up formatting for the movie files
        Writer = animation.writers[out_writer]
        writer = Writer(fps=15, metadata=dict(artist="animationsMD1D_David_Hahn_Benjamin_Schroeder"), bitrate=1800)
        ani.save(out_path, writer=writer, dpi=dpi)

    return ani, out_path
