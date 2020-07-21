"""
    This style file is used to control the styling of the visualizations followed
"""

import matplotlib as mpl 
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy import constants



def cm2inch(value:float):
    return value/2.54


#GENERAL SETTINGS
dpi=300
dpi_animation=100

#COLORS:
alpha_val = 1
alpha_traj = 0.8
aplha_light=0.3

traj_current = "k"
traj_start = "blue"
traj_end = "red"
trajectory_color = "orange"
animation_traj = cm.get_cmap("inferno")

potential_color = cm.get_cmap("tab10")
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
