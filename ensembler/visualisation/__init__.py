"""
A collection of visualizations (plots, animation, interactive widgets)
"""
import matplotlib
from scipy import constants
from IPython import get_ipython


def cm2inch(value: float):
    return value / 2.54


# settings:
dpi = 300
dpi_animation = 100

if "zmq" in str(type(get_ipython())):
    figsize = [12, 7]
    animation_figsize = [9, 6]
    figsize_doubleColumn = [16, 9]

    # fontSizes
    ax_title = 26
    ax_label = 24
    ax_tick = 14
    ax_legend = 16
    font_gen = 16

else:
    figsize = [cm2inch(8.6), cm2inch(8.6 / constants.golden)]
    animation_figsize = figsize
    figsize_doubleColumn = [cm2inch(2 * 8.6), cm2inch(2 * 8.6 / constants.golden)]

    # fontSizes
    ax_title = 14
    ax_label = 12
    ax_tick = 10
    ax_legend = 10
    font_gen = 12


plot_layout_settings = {
    "font.family": "sans-serif",
    "font.serif": "Times",
    "font.size": font_gen,
    "xtick.labelsize": ax_tick,
    "ytick.labelsize": ax_tick,
    "axes.labelsize": ax_label,
    "axes.titlesize": ax_title,
    "legend.fontsize": ax_legend,
    "savefig.dpi": dpi,
    "figure.figsize": figsize,
    "figure.facecolor": "white",
    "animation.html": "jshtml",
}

for key, value in plot_layout_settings.items():
    matplotlib.rcParams[key] = value
