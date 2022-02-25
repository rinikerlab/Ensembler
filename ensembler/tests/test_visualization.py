import numpy as np
import unittest

from ensembler.potentials.OneD import harmonicOscillatorPotential
from ensembler.potentials.TwoD import harmonicOscillatorPotential as harmonicOscillatorPotential2D
from ensembler.samplers.stochastic import metropolisMonteCarloIntegrator
from ensembler.system import system


class test_Visualization(unittest.TestCase):
    pass


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


class test_plot_Simulations(test_Visualization):
    def test_static_sim_plots(self):
        sim = system(potential=harmonicOscillatorPotential(), sampler=metropolisMonteCarloIntegrator())
        sim.simulate(100)
        plotSimulations.oneD_simulation_analysis_plot(sim)

    """
    def test_static_sim_bias_plots(self):
        sim = system(potential=addedPotentials(), sampler=metropolisMonteCarloIntegrator())
        sim.simulate(100)
        plotSimulations.oneD_biased_simulation_analysis_plot(sim)
    """

    def test_twoD_simulation_ana_plot(self):
        # settings
        sim_steps = 100
        pot2D = harmonicOscillatorPotential2D()
        sampler = metropolisMonteCarloIntegrator()
        sys = system(potential=pot2D, sampler=sampler, start_position=[0, 0])

        # simulate
        cur_state = sys.simulate(sim_steps, withdraw_traj=True)

        plotSimulations.twoD_simulation_analysis_plot(system=sys)


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
        sim = system(potential=harmonicOscillatorPotential(), sampler=metropolisMonteCarloIntegrator())
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
