#imports
import numpy as np
from ensembler.potentials import OneD as potentials1D
from ensembler.visualisation import plotPotentials

#phase space
positions = np.linspace(start=0, stop=10, num=1000)

#build potential
V = potentials1D.fourWellPotential(a=1,b=4, c=6, d=8)

#visualize
fig, outpath = plotPotentials.plot_1DPotential_Term(potential=V, positions=positions, out_path="four_well.png")



#Langevin integration simulation:
##imports
from ensembler.potentials.OneD import harmonicOscillatorPotential
from ensembler.integrator.stochastic import langevinIntegrator
from ensembler.system import system
from ensembler.visualisation.plotSimulations import static_sim_plots

##settings
sim_steps = 5000
time_step = 0.1
start_position = 0
temperature = 300

##Simulation Setup
pot=harmonicOscillatorPotential()
integrator = langevinIntegrator(dt=time_step)
sys=system(potential=pot, integrator=integrator,  position=start_position,  temperature=temperature)

##simulate
cur_state = sys.simulate(sim_steps)

##visualize
fig, out_path = static_sim_plots(sys, title="Langevin Simulation", out_path="langevine_simulation.png")