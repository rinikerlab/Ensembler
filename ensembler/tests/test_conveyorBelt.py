import unittest
from ensembler.system.basic_system import system
from ensembler.system.perturbed_system import perturbedSystem
from ensembler.ensemble.replicas_dynamic_parameters import ConveyorBelt
from ensembler.integrator.stochastic import metropolisMonteCarloIntegrator, monteCarloIntegrator
from ensembler.potentials import OneD

class testEnsemble(unittest.TestCase):
    convBelt = ConveyorBelt

    def test_constructor(self):
        ens = self.convBelt(0, 2)

    def test_run_step_lambda2(self):
        integrator = monteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)

        lam = 0.5
        sys.set_lam(lam)
        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def test_run_step(self):
        integrator = monteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)

        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def test_run_step_lambda1(self):
        integrator = monteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)

        lam = 1.0
        self.sys.set_lam(lam)
        ens = self.convBelt(0.0, 1, system=sys)

        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def test_run_step_lambda2(self):
        integrator = monteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)

        lam = 0.5
        sys.set_lam(lam)
        ens = self.convBelt(0.0, 1, system=sys)
        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def testTraj(self):
        integrator = monteCarloIntegrator()
        ha = OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = OneD.harmonicOscillatorPotential(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)
        sys = perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)

        lam = 0.5
        sys.set_lam(lam)
        ens = self.convBelt(0.0, 1, system=sys)

        # print(ens.run(())
        # ens = ensemble.ConveyorBelt(0.0, 8, system=sys)
        #
        #ensemble.calc_traj_file(steps=100, ens=ens)
        #import os
        #os.remove(os.getcwd()+"/traj_*.dat")

if __name__ == '__main__':
    unittest.main()
