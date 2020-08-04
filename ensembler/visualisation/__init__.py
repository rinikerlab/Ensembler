import matplotlib
from matplotlib import cm
from scipy import constants

def cm2inch(value:float):
    return value/2.54

#settings:
figsize=[cm2inch(8.6), cm2inch(8.6/constants.golden)]
figsize_doubleColumn =[cm2inch(2*8.6), cm2inch(2*8.6/constants.golden)]


plot_layout_settings = {'font.family': 'sans-serif',
                        "font.serif": 'Times',
                        "font.size":10,
                        'xtick.labelsize': 6,
                        'ytick.labelsize': 6,
                        'axes.labelsize': 10,
                        'axes.titlesize': 12,
                        'legend.fontsize': 6,
                        'savefig.dpi': 600,
                        'figure.figsize': figsize,
                        }


for key, value in plot_layout_settings.items():
    matplotlib.rcParams[key] = value
