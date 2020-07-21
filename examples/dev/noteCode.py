#imports
import numpy as np
from ensembler.potentials import OneD as potentials1D
from ensembler.visualisation.plotPotentials import plot_1DPotential_Term

#phase space
positions = np.linspace(start=0, stop=11, num=1000)

#build potential
V = potentials1D.fourWellPotential(a=2,b=5, c=7, d=9)

#visualize
fig, outpath = plot_1DPotential_Term(potential=V, positions=positions, out_path="four_well.png")





#Langevin integration simulation:
##imports
from ensembler.potentials.OneD import harmonicOscillatorPotential
from ensembler.integrator.stochastic import langevinIntegrator
from ensembler.system import system
from ensembler.visualisation.plotSimulations import static_sim_plots

##Simulation Setup
pot=harmonicOscillatorPotential()
integrator = langevinIntegrator(dt=0.1)
sys=system(potential=pot, integrator=integrator,  position=0,  temperature=300)

##simulate
cur_state = sys.simulate(steps=10000)

##visualize
fig, out_path = static_sim_plots(sys, title="Langevin Simulation", out_path="langevine_simulation.png")