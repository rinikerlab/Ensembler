import matplotlib
from scipy import constants


def cm2inch(value: float):
    return value / 2.54


# settings:
figsize = [cm2inch(8.6), cm2inch(8.6 / constants.golden)]
figsize_doubleColumn = [cm2inch(2 * 8.6), cm2inch(2 * 8.6 / constants.golden)]

plot_layout_settings = {'font.family': 'sans-serif',
                        "font.serif": 'Times',
                        "font.size": 10,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'axes.labelsize': 12,
                        'axes.titlesize': 14,
                        'legend.fontsize': 8,
                        'savefig.dpi': 600,
                        'figure.figsize': figsize,
                        }

for key, value in plot_layout_settings.items():
    matplotlib.rcParams[key] = value
