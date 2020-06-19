"""
    [summary]
"""

import matplotlib as mpl 
from matplotlib import cm
from matplotlib import pyplot as plt



#GENERAL SETTINGS
dpi=300
dpi_animation=100
figsize = (16, 9)

#FONTS:
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#COLORS:
alpha_val = 1
alpha_traj = 0.8
aplha_light=0.3

traj_current = "k"
traj_start = "green"
traj_end = "red"
trajectory_color = "orange"
animation_traj = cm.get_cmap("inferno")

potential_color = plt.get_cmap("tab10")
potential_light = "grey"
potential_resolution=1000

##Maps:
###1 information dimension - binary information
gradient_green_map = cm.get_cmap("Greens")
gradient_reds_map = cm.get_cmap("Reds")
gradient_blue_map = cm.get_cmap("Blues")

###1 information dimension - gradient
gradient_map = cm.get_cmap("viridis")

###gradient map, optimal is centered:
gradient_centered_map = cm.get_cmap("gist_earth_r")

###qualitative info - mutli step recognition
qualitative_map = cm.get_cmap("tab20b")
