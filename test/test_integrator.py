import unittest
import numpy as np
from ensembler import integrator as integ, potentials as pot, system

class test_Integrators(unittest.TestCase):
    pass

class test_MonteCarlo_Integrator(unittest.TestCase):
    def test_constructor(self):
        integrator = integ.monteCarloIntegrator()

    def test_step(self):
        potent = pot.OneD.harmonicOsc()
        integrator = integ.monteCarloIntegrator()
        sys = system.system(potential=potent, integrator=integrator)

        old_pos, oldForce = sys._currentPosition, sys._currentForce
        newPos, _, posShift= integrator.step(system=sys)

        self.assertNotEqual(old_pos, newPos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, posShift, msg="Nothing happened here!")
        pass

    def test_integrate(self):
        potent = pot.OneD.harmonicOsc()
        integrator = integ.monteCarloIntegrator()
        steps=42
        sys = system.system(potential=potent, integrator=integrator)
        sys.trajectory = []
        old_pos, oldForce = sys._currentPosition, sys._currentForce
        integrator.integrate(system=sys, steps=steps)
        new_pos, new_Force = sys._currentPosition, sys._currentForce
        self.assertEqual(steps, len(sys.trajectory), msg="The simulation did not run or was too short!")
        self.assertNotEqual(old_pos, new_pos, msg="Nothing happened here!")
        self.assertNotEqual(oldForce, new_Force, msg="Nothing happened here!")
        pass

class test_MetropolisMonteCarlo_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.metropolisMonteCarloIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_verlocityVerletIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.velocityVerletIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_positionVerletIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.positionVerletIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_leapFrogIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.leapFrogIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

if __name__ == '__main__':
    unittest.main()
