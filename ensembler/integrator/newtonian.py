"""
Newtonian Integrators
"""

import numpy as np
from typing import Tuple
import scipy.constants as const

from ensembler import system
from ensembler.integrator._basicIntegrators import _integratorCls


class newtonianIntegrator(_integratorCls):
    currentPosition:float
    currentVelocity:float
    currentForces:float

    dt:float


class velocityVerletIntegrator(newtonianIntegrator):
    """
        .. autoclass:: Verlocity Verlet Integrator,
        is not implemented yet
    """
    def __init__(self, dt=0.0005):
        self.dt = dt

    pass


class positionVerletIntegrator(newtonianIntegrator):

    def __init__(self, dt=0.0005):
        self.dt = dt

    def step(self, system):
        #init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities

        #calculation:
        newForces = system.potential.dvdpos(currentPosition)  #Todo: make multi particles possible - use current forces!
        new_velocity = currentVelocity - (newForces / (system.mass))
        new_position = currentPosition+ new_velocity * self.dt

        if(self.verbose):
            print("INTEGRATOR: current forces\t ", newForces)
            print("INTEGRATOR: current Velocities\t ", currentVelocity)
            print("INTEGRATOR: current_position\t ", currentPosition)

            print("INTEGRATOR: newVel\t ", new_velocity)
            print("INTEGRATOR: newPosition\t ", new_position)
            print("\n")
        return new_position, new_velocity, newForces


class leapFrogIntegrator(newtonianIntegrator):
    def __init__(self, dt=0.0005):
        self.dt = dt

    def step(self, system):
        raise Exception("not implemented!")
        pass



"""
OLD Integrators:
class newtonIntegrator(integrator): #LEAPFROG
    def step(self, sys):
        sys.pos = sys.newpos  # t
        sys.force = -sys.pot.dhdpos(sys.lam, sys.pos)  # t
        sys.oldvel = sys.new_velocity  # t - 0.5 Dt
        sys.new_velocity += sys.force / sys.mu * self.dt  # t+0.5Dt
        sys.vel = 0.5 * (sys.oldvel + sys.new_velocity)
        sys.newpos += self.dt * sys.new_velocity  # t+Dt
        sys.veltemp = sys.mu / const.gas_constant / 1000.0 * sys.vel ** 2  # t
        sys.updateEne()
        return [sys.pos, sys.vel, sys.veltemp, sys.totkin, sys.totpot, sys.totene, sys.lam, sys.dhdlam]

    def __init__(self, dt=1e-1):
        self.dt = dt
        raise NotImplementedError("This "+__class__+" class is not implemented")

class nhIntegrator(integrator): Nosehover-leapfrog
    def scaleVel(self, sys):
        freetemp = 2.0 / const.gas_constant / 1000.0 * sys.mu * sys.new_velocity ** 2  # t+0.5Dt
        self.oldxi = self.xi  # t-0.5Dt
        self.xi += self.dt / (self.tau * self.tau) * (freetemp / sys.temp - 1.0)  # t+0.5t
        scale = 1.0 - self.xi * self.dt
        return scale

    def step(self, sys):
        sys.pos = sys.newpos  # t
        sys.force = -sys.pot.dhdpos(sys.lam, sys.pos)  # t
        sys.oldvel = sys.new_velocity  # t - 0.5 Dt
        sys.new_velocity += sys.force / sys.mu * self.dt  # t+0.5t
        sys.new_velocity *= self.scaleVel(sys)
        sys.vel = 0.5 * (sys.oldvel + sys.new_velocity)
        sys.newpos += self.dt * sys.new_velocity  # t+Dt
        sys.veltemp = sys.mu / const.gas_constant/1000.0 * sys.vel ** 2  # t
        sys.updateEne()
        return [sys.pos, sys.vel, sys.veltemp, sys.totkin, sys.totpot, sys.totene, sys.lam, sys.dhdlam]

    def __init__(self, xi=0.0, tau=0.1, dt=1e-1):
        self.xi = xi
        self.oldxi = xi
        self.tau = tau
        self.dt = dt
        raise NotImplementedError("This "+__class__+" class is not implemented")

class hmcIntegrator(integrator):
    def step(self, sys):
        accept = 0
        oldene = sys.totene
        oldpos = sys.pos  # t
        oldvel = sys.vel
        sys.initVel()  # t-0.5Dt
        for i in range(self.steps):
            sys.pos += self.dt * sys.new_velocity  # t
            force = -sys.pot.dhdpos(sys.lam, sys.pos)  # t
            sys.vel = (oldvel + sys.new_velocity) / 2.0
            sys.veltemp = sys.mu / const.gas_constant / 1000.0 * sys.vel ** 2  # t
            sys.updateEne()
            oldvel = sys.new_velocity
            sys.new_velocity += force / sys.mu * self.dt  # t+0.5t
        accept = 0
        if sys.totene < oldene:
            accept = 1
        else:
            if np.random.rand() <= np.exp(-1 / (const.gas_constant / 1000.0* sys.temp) * (sys.totene - oldene)):
                accept = 1
        if accept == 0:
            sys.pos = oldpos
            sys.vel = oldvel
            sys.veltemp = sys.mu / const.gas_constant * sys.vel ** 2  # t
            sys.updateEne()
        return [sys.pos, sys.vel, sys.veltemp, sys.totkin, sys.totpot, sys.totene, sys.lam, sys.dhdlam]

    def __init__(self, steps=5, dt=1e-1):
        self.steps = steps
        self.dt = dt
        raise NotImplementedError("This "+__class__+" class is not implemented")
"""
