import os, unittest, tempfile
from ensembler import integrator as integ, potentials as pot, system
from ensembler.integrator import _basicIntegrators
from ensembler.integrator import stochastic, newtonian, optimizers

"""
STOCHASTIC INTEGRATORS
"""
tmp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix="test_integrators")


class standard_IntegratorTests(unittest.TestCase):
    integrator_class = _basicIntegrators._integratorCls
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

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
        print(integrator)


class test_MonteCarlo_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.monteCarloIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")
        pass


class test_MetropolisMonteCarlo_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.metropolisMonteCarloIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_Langevin_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.langevinIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_LangevinVelocity_Integrator(standard_IntegratorTests):
    integrator_class = stochastic.langevinVelocityIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


"""
NETOWNIAN
"""


class test_verlocityVerletIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.velocityVerletIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_positionVerletIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.positionVerletIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


class test_leapFrogIntegrator_Integrator(standard_IntegratorTests):
    integrator_class = newtonian.leapFrogIntegrator
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

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
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + integrator_class.name + "_", suffix=".obj", dir=tmp_dir)

    def test_step(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift = integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")

    def test_integrate(self):
        potent = pot.OneD.harmonicOscillatorPotential()
        integrator = self.integrator_class()

        steps = 42
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce

        self.assertEqual(steps + 1, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")


if __name__ == '__main__':
    unittest.main()
