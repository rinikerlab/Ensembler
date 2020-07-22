import unittest
from ensembler.system.basic_system import system
from ensembler.ensemble.replicas_dynamic_parameters import ConveyorBelt
from ensembler.integrator.stochastic import stochasticIntegrator
from ensembler.potentials import OneD

class testEnsemble(unittest.TestCase):
    def testEnsemble(self):
        ens = ConveyorBelt(0.0, 1)
        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def testEnsembleSystem(self):
        integrator = stochasticIntegrator()
        ha = OneD.harmonicOscillator(x_shift=-5)
        hb = OneD.harmonicOscillator(x_shift=5)
        pot = OneD.linearCoupledPotentials(Va=ha, Vb=hb)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
        ens = ConveyorBelt(0.0, 1, system=sys)

        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def testEnsembleSystemShift(self):
        integrator = integ.metropolisMonteCarloIntegrator()
        ha = potent.OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = potent.OneD.harmonicOscillatorPotential(x_shift=5)
        lam = 0.5
        pot = potent.OneD.linearCoupledPotentials(Va=ha, Vb=hb, lam=lam)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
        ens = ensemble.ConveyorBelt(0.0, 1, system=sys)
        ens.calculate_conveyorBelt_totEne()
        ens.run()
        ens.calculate_conveyorBelt_totEne()
        ens.get_replicas_positions()

    def testTraj(self):
        integrator = integ.metropolisMonteCarloIntegrator()
        ha = potent.OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = potent.OneD.harmonicOscillatorPotential(x_shift=5)
        lam = 0.5
        pot = potent.OneD.linearCoupledPotentials(Va=ha, Vb=hb, lam=lam)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
        ens = ensemble.ConveyorBelt(0.0, 1, system=sys)

        # print(ens.run(())
        # ens = ensemble.ConveyorBelt(0.0, 8, system=sys)
        #
        #ensemble.calc_traj_file(steps=100, ens=ens)
        #import os
        #os.remove(os.getcwd()+"/traj_*.dat")

if __name__ == '__main__':
    unittest.main()
