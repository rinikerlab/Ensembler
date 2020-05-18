import numpy as np
from typing import Iterable
from numbers import Number
from matplotlib import animation, pyplot as plt

import os, sys
sys.path.append(os.path.dirname(__file__)+"/..")

from ensembler.system import system

def animation_trajectory(sys: system, x_range=None, y_range=None, title:str=None, out_path:str=None, out_writer:str="pillow", dpi:int=100)-> (animation.Animation, (str or None)):
    # plotting
    x1data = sys.trajectory.position
    y1data = sys.trajectory.totPotEnergy
    shift = sys.trajectory.dhdpos

    x_max = max(x1data)
    x_min = min(x1data)

    if (x_range == None):
        xtot_space = np.arange(x_min + 0.2 * x_min, x_max + 0.2 * x_max + 1)
    else:
        xtot_space = np.arange(min(x_range), max(x_range) + 1, 1)
    5
    ytot_space = sys.potential.ene(xtot_space)

    tmax = len(y1data) - 1
    t0 = 0
    step_size = 1

    active_dots = 20

    xdata, ydata = [], []
    # figures
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xtot_space, ytot_space, label="wholePot")
    # data structures in ani
    line, = ax.plot([], [], c="blue", alpha=0.3, lw=2)
    line.set_data(xtot_space, ytot_space)

    scatter = ax.scatter([], [], c=[], vmin=0, vmax=1, cmap='inferno')  # , cmap=cm.viridis)#todo: FIND NICE COLORMAP
    start_p, = ax.plot([], [], "bo", c="g", ms=10)
    end_p, = ax.plot([], [], "bo", c="r", ms=10)
    curr_p, = ax.plot([], [], "bo", c="k", ms=10)

    # Params
    ax.set_xlabel("$r$")
    ax.set_ylabel("$V$")
    if (title != None):
        fig.suptitle(title)

    def init():
        del xdata[:], ydata[:]

        end_p.set_data([], [])
        curr_p.set_data([], [])

        if (x_range != None):
            ax.set_xlim(x_range)
        if (y_range != None):
            ax.set_ylim(y_range)
        # return line,

    def data_gen(t=t0):
        while t < tmax:
            t += step_size
            yield x1data[t], y1data[t]

    def run(data):
        # update the data
        x, V = data
        lastindx = x1data.shape[-1]-1
        lastindy = np.array(list(y1data)).shape[-1]-1

        if ( x == x1data[lastindx]): #x.all(x1data[-1])):  # last step of traj
            curr_p.set_data([], [])
            end_p.set_data(x1data[lastindx], y1data[lastindy])
        else:
            curr_p.set_data([x], [V])
            xdata.append(x)
            ydata.append(V)

            if(len(xdata)>active_dots):
                c=np.concatenate((np.array([0.6 for x in range(len(xdata)-active_dots)]), np.linspace(0.6, 0, active_dots)))
            else:
                c=np.linspace(0.6, 0, len(xdata))

            scatter.set_offsets(np.c_[xdata, ydata])
            scatter.set_array(c)

        if (min(ax.get_ylim()) > V):
            ax.set_ylim(V + V * 0.1, max(ax.get_ylim()))
        return line

    ani = animation.FuncAnimation(fig=fig, func=run, frames=data_gen, init_func=init, blit=False,
                                  interval=20, repeat=False, save_count=len(x1data))
    if (out_path != None):
        # Set up formatting for the movie files
        Writer = animation.writers[out_writer]
        writer = Writer(fps=15, metadata=dict(artist='animationsMD1D_David_Hahn_Benjamin_Schroeder'), bitrate=1800)
        ani.save(out_path, writer=writer, dpi=dpi)

    return ani, out_path

def animation_EDS_trajectory(system: system, x_range=None, title:str=None, out_path:str=None, hide_legend:bool=True,
                             s_values:list=[1.0], step_size:float=1, out_writer:str="pillow", dpi:int=100, tot_pot_resolution:int=100)-> (animation.Animation, (str or None)):
    # plotting
    x1data = np.array(system.trajectory.position)
    y1data = system.trajectory.totPotEnergy
 
    x_max = max(x1data)
    x_min = min(x1data)
    active_dots = 20

    if (x_range == None):
        xtot_space = np.array(np.arange(x_min + 0.2 * x_min, x_max + 0.2 * x_max + 1), ndmin=1)
    else:
        xtot_space = np.array(np.linspace(min(x_range), max(x_range) + 1, tot_pot_resolution), ndmin=1)

    tmax = len(y1data) - 1-step_size
    t0 = 0

    xdata, ydata = [], []

    # figures
    fig = plt.figure()
    ax = fig.add_subplot(111)

    from ensembler.visualisation.plotPotentials import envPot_differentS_overlay_plot

    _, ax = envPot_differentS_overlay_plot(eds_potential=system.potential, s_values=s_values, title=title,
                                           positions=xtot_space, axes=ax, hide_legend=hide_legend)

    scatter = ax.scatter([], [], c=[], vmin=0, vmax=1, cmap='inferno')  # , cmap=cm.viridis)#todo: FIND NICE COLORMAP
    start_p, = ax.plot([], [], "bo", c="g", ms=10)
    end_p, = ax.plot([], [], "bo", c="r", ms=10)
    curr_p, = ax.plot([], [], "bo", c="k", ms=10)

    # Params
    ax.set_xlabel("$r$")
    ax.set_ylabel("$V$")
    if (isinstance(title, type(None))):
        fig.suptitle(title)

    def init():
        del xdata[:], ydata[:]

        end_p.set_data([], [])
        curr_p.set_data([], [])

        if (x_range != None):
            ax.set_xlim(x_range)
        # return line,

    def data_gen(t=t0):
        while t < tmax:
            t += step_size
            yield x1data[t], y1data[t]

    def run(data):
        # update the data
        x, V = data
        if (type(x) == type(x1data[-1])==list and all([xi == x1i for xi, x1i in zip(x, x1data[-1])]) ):  # last step of traj
            curr_p.set_data([], [])
            end_p.set_data(x1data[-1], y1data[-1])
        elif (type(x) == type(x1data[-1])==Number and x==x1data[-1] ):  # last step of traj
            curr_p.set_data([], [])
            end_p.set_data(x1data[-1], y1data[-1])
        else:
            curr_p.set_data([x], [V])
            xdata.append(x) if(not isinstance(x, Iterable)) else xdata.append(x[0])
            ydata.append(V)

            if (len(xdata) > active_dots+10):
                c = np.concatenate((np.array([0.6 for x in range(len(xdata) - active_dots)]), np.linspace(0.6, 0, active_dots)))
            else:
                c = np.linspace(0.6, 0, len(xdata))

            scatter.set_offsets(np.c_[xdata, ydata])
            scatter.set_array(c)

        if (min(ax.get_ylim()) > V):
            ax.set_ylim(V + V * 0.1, max(ax.get_ylim()))

    ani = animation.FuncAnimation(fig=fig, func=run, frames=data_gen, init_func=init, blit=False,
                                  interval=20, repeat=False, save_count=len(x1data))
    if (out_path != None):
        # Set up formatting for the movie files
        Writer = animation.writers[out_writer]
        writer = Writer(fps=15, metadata=dict(artist='animationsMD1D_David_Hahn_Benjamin_Schroeder'), bitrate=1800)
        ani.save(out_path, writer=writer, dpi=dpi)

    return ani, out_path
