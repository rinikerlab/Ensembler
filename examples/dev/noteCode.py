# Four Well Potential
##Imports
import numpy as np
from ensembler.potentials.OneD import fourWellPotential
from ensembler.visualisation.plotPotentials import plot_1DPotential

##Build potential
V = fourWellPotential(Vmax=4, a=1.5, b=4.0, c=7.0, d=9.0, ah=2.0, bh=0.0, ch=0.5, dh=1.0)

##Visualize
positions = np.linspace(start=0, stop=11, num=1000)  # phase space to be visualized
fig, outpath = plot_1DPotential(potential=V, positions=positions, title="Four Well Potential")


# Langevin integration simulation:
##Imports
from ensembler.potentials.OneD import fourWellPotential
from ensembler.samplers.stochastic import langevinIntegrator
from ensembler.system import system
from ensembler.visualisation.plotSimulations import oneD_simulation_analysis_plot

##Simulation Setup
V = fourWellPotential(Vmax=4, a=1.5, b=4.0, c=7.0, d=9.0, ah=2.0, bh=0.0, ch=0.5, dh=1.0)
sampler = langevinIntegrator(dt=0.1, gamma=10)
sys = system(potential=V, sampler=sampler, start_position=4, temperature=1)

##Simulate
sys.simulate(steps=1000)

##Visualize
positions = np.linspace(start=0, stop=10, num=1000)  # phase space to be visualized
oneD_simulation_analysis_plot(system=sys, title="Langevin Simulation", limits_coordinate_space=positions)


# Local elevation/metadynamics simulation:
##Imports
from ensembler.potentials.OneD import fourWellPotential, metadynamicsPotential
from ensembler.samplers.stochastic import langevinIntegrator
from ensembler.system import system
from ensembler.visualisation.plotSimulations import oneD_simulation_analysis_plot

##Simulation Setup
origpot = fourWellPotential(Vmax=4, a=1.5, b=4.0, c=7.0, d=9.0, ah=2.0, bh=0.0, ch=0.5, dh=1.0)
V = metadynamicsPotential(origpot, amplitude=0.5, sigma=0.2, n_trigger=5)

sampler = langevinIntegrator(dt=0.1, gamma=10)

sys = system(potential=V, sampler=sampler, start_position=4, temperature=1)

##Simulate
sys.simulate(steps=1000)

##Visualize
positions = np.linspace(start=0, stop=10, num=1000)  # phase space to be visualized
oneD_simulation_analysis_plot(system=sys, title="Local Elevation/Metadynamics Simulation", limits_coordinate_space=positions)
