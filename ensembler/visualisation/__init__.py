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
                        'xtick.labelsize': 7,
                        'ytick.labelsize': 7,
                        'axes.labelsize': 9,
                        'axes.titlesize': 11,
                        'legend.fontsize': 10,
                        'savefig.dpi': 900,
                    }

for key, value in plot_layout_settings.items():
    matplotlib.rcParams[key] = value
