import os
import tempfile
import unittest

from ensembler import potentials as pot, system
from ensembler.samplers import _basicSamplers
from ensembler.samplers import stochastic, newtonian, optimizers

"""
STOCHASTIC INTEGRATORS
"""


class standard_IntegratorTests(unittest.TestCase):
    integrator_class = _basicSamplers._samplerCls
    tmp_test_dir: str = None

    def setUp(self) -> None:
        test_dir = os.getcwd() + "/tests_out"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if __class__.tmp_test_dir is None:
            __class__.tmp_test_dir = tempfile.mkdtemp(dir=test_dir, prefix="tmp_test_sampler")
        _, self.tmp_out_path = tempfile.mkstemp(prefix="test_" + self.integrator_class.name, suffix=".obj", dir=__class__.tmp_test_dir)

    def test_constructor(self):
        integrator = self.integrator_class()

    def test_save_integrator(self):
        integrator = self.integrator_class()
        integrator.save(self.tmp_out_path)

    def test_load_integrator(self):
        integrator = self.integrator_class()
        integrator.save(self.tmp_out_path)
        del integrator

        integrator = self.integrator_class.load(self.tmp_out_path)
        # print(integrator)


class test_MonteCarlo_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.metropolisMonteCarloIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")
        pass


class test_MetropolisMonteCarlo_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.metropolisMonteCarloIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_Langevin_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.langevinIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_LangevinVelocity_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.langevinVelocityIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        # print(sys.trajectory)

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


"""
NETOWNIAN
"""


class test_verlocityVerletIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.velocityVerletIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_positionVerletIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.positionVerletIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_leapFrogIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.leapFrogIntegrator

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


"""
Optimizer
"""


class test_cg_Integrator(standard_IntegratorTests):
    integrator_class = optimizers.conjugate_gradient

    def test_step(self):
        position = 1
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, sampler=integrator, start_position=position)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")

    def test_integrate(self):
        position = 1
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, sampler=integrator, start_position=position)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


if __name__ == "__main__":
    unittest.main()
