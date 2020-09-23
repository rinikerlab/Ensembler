import numpy as np
import os
import tempfile
import unittest

from ensembler.potentials.OneD import harmonicOscillatorPotential
from ensembler.samplers.stochastic import monteCarloIntegrator
from ensembler.system import system

class test_Visualization(unittest.TestCase):
    tmp_test_dir: str = None


    def setUp(self) -> None:
        pass
       #if(__class__.tmp_test_dir is None):
       #     __class__.tmp_test_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix="tmp_test_visualization_Potentials")

"""
   Potential Plots
"""
from ensembler.visualisation import plotPotentials
class test_plot_Potentials(test_Visualization):

    def test_plot_1DPotential(self):
        plotPotentials.plot_1DPotential(harmonicOscillatorPotential(), np.linspace(-10, 10, 100))

    def test_plot_1DPotential_dhdpos(self):
        plotPotentials.plot_1DPotential_dhdpos(harmonicOscillatorPotential(), np.linspace(-10, 10, 100))

    def test_1DPotential_V(self):
        plotPotentials.plot_1DPotential_V(harmonicOscillatorPotential(), np.linspace(-10, 10, 100))

    def test_1DPotential_V(self):
        plotPotentials.plot_1DPotential_Termoverlay(harmonicOscillatorPotential(), np.linspace(-10, 10, 100))

"""
   Simulation
"""
from ensembler.visualisation import plotSimulations
from ensembler.potentials.biased_potentials.biasOneD import addedPotentials
class test_plot_Simulations(test_Visualization):

    def test_static_sim_plots(self):
        sim = system(potential=harmonicOscillatorPotential(), sampler=monteCarloIntegrator())
        sim.simulate(100)
        plotSimulations.static_sim_plots(sim)

    def test_static_sim_bias_plots(self):
        sim = system(potential=addedPotentials(), sampler=monteCarloIntegrator())
        sim.simulate(100)
        plotSimulations.static_sim_plots_bias(sim)


"""
   Conveyor Belt
"""
from ensembler.visualisation import plotConveyorBelt
class test_plot_ConveyorBelt(test_Visualization):

    def test_1D_plotEnsembler(self):
        lam = np.linspace(0, 1, 10)
        lam = np.linspace(0, 1, 10)
        plotConveyorBelt.plotEnsembler(lam, lam)


"""
   Animation
"""
from ensembler.visualisation import animationSimulation
class test_plot_Animations(test_Visualization):

    def test_1D_animation(self):
        sim = system(potential=harmonicOscillatorPotential(), sampler=monteCarloIntegrator())
        sim.simulate(100)
        animationSimulation.animation_trajectory(simulated_system=sim)

"""
Interactive Widgets
"""
from ensembler.visualisation import interactive_plots
class test_interactive_widgets(test_Visualization):
    def test_1D_cvb(self):
        interactive_plots.interactive_conveyor_belt(nbins=10, numsys=3, steps=10)

    def test_1D_eds(self):
        interactive_plots.interactive_eds()