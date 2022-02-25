import unittest

from ensembler.ensemble.replicas_dynamic_parameters import conveyorBelt
from ensembler.samplers.stochastic import metropolisMonteCarloIntegrator
from ensembler.potentials import OneD
from ensembler.system.perturbed_system import perturbedSystem


class testEnsemble(unittest.TestCase):
    convBelt = conveyorBelt

    def test_constructor(self):
        ens = self.convBelt(0, 2)

    def test_run_step_lambda2(self):
        integrator = metropolisMonteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, sampler=integrator)

        lam = 0.5
        sys.lam = lam
        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_total_ensemble_energy()
        ens.run()
        ens.calculate_total_ensemble_energy()
        ens.get_replicas_positions()

    def test_run_step(self):
        integrator = metropolisMonteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, sampler=integrator)

        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_total_ensemble_energy()
        ens.run()
        ens.calculate_total_ensemble_energy()
        ens.get_replicas_positions()

    def test_run_step_lambda1(self):
        integrator = metropolisMonteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, sampler=integrator)

        lam = 1.0
        sys.lam = lam
        ens = self.convBelt(0.0, 1, system=sys)

        ens.calculate_total_ensemble_energy()
        ens.run()
        ens.calculate_total_ensemble_energy()
        ens.get_replicas_positions()

    def test_run_step_lambda2(self):
        integrator = metropolisMonteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, sampler=integrator)

        lam = 0.5
        sys.lam = lam
        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_total_ensemble_energy()
        ens.run()
        ens.calculate_total_ensemble_energy()
        ens.get_replicas_positions()

    def testTraj(self):
        integrator = metropolisMonteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, sampler=integrator)

        lam = 0.5
        sys.lam = lam
        ens = self.convBelt(0.0, 1, system=sys)

        # print(ens.run(())
        # ens = ensemble.ConveyorBelt(0.0, 8, system=sys)
        #
        # ensemble.calc_traj_file(steps=100, ens=ens)
        # import os
        # os.remove(os.getcwd()+"/traj_*.dat")


if __name__ == "__main__":
    unittest.main()
