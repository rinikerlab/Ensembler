"""
Newtonian Integrators
"""

from ensembler.integrator._basicIntegrators import _integratorCls
from ensembler.util.ensemblerTypes import Union
from ensembler.util.ensemblerTypes import system as systemType


class newtonianIntegrator(_integratorCls):
    """
    newtonianIntegrator [summary]

    """
    currentPosition: float
    currentVelocity: float
    currentForces: float

    dt: float

    def __init__(self, dt=0.0005):
        self.dt = dt


class velocityVerletIntegrator(newtonianIntegrator):
    """
    velocityVerletIntegrator [summary]
    
    Verlet, Loup (1967). "Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard−Jones Molecules". Physical Review. 159 (1): 98–103.
    """
    name = "Verlocity Verlet Integrator"

    def step(self, system: systemType) -> Union[float, float, float]:
        """
        step [summary]

        Parameters
        ----------
        system : systemType
            [description]
        """
        # init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities
        currentForces = system._currentForce

        # calculation:
        new_position = currentPosition + currentVelocity * self.dt - (
                    (0.5 * currentForces * (self.dt ** 2)) / system.mass)
        new_forces = system.potential.force(new_position)
        new_velocity = currentVelocity - ((0.5 * (currentForces + new_forces) * self.dt) / system.mass)

        if (self.verbose):
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")

        return new_position, new_velocity, new_forces


class positionVerletIntegrator(newtonianIntegrator):
    name = "Position Verlet Integrator"

    def step(self, system: systemType) -> Union[float, float, float]:
        """
        step [summary]

        Parameters
        ----------
        system : systemType
            [description]

        Returns
        -------
        Union[float, float, float]
            [description]
        """
        # init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities

        # calculation:
        new_forces = system.potential.force(currentPosition)
        new_velocity = currentVelocity - (new_forces / system.mass)
        new_position = currentPosition + new_velocity * self.dt

        if (self.verbose):
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")
        return new_position, new_velocity, new_forces


class leapFrogIntegrator(newtonianIntegrator):
    """
    leapFrogIntegrator [summary]


    """
    name = "Leap Frog Integrator"

    def step(self, system: systemType) -> Union[float, float, float]:
        """
        step [summary]

        Parameters
        ----------
        system : systemType
            [description]

        Returns
        -------
        Union[float, float, float]
            [description]
        """

        # init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities
        currentForces = system._currentForce

        # calculation:
        v_halft = currentVelocity - ((0.5 * self.dt * currentForces) / system.mass)
        new_position = currentPosition + v_halft * self.dt
        new_forces = system.potential.force(new_position)
        new_velocity = v_halft - ((0.5 * new_forces * self.dt) / system.mass)

        if (self.verbose):
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")

        return new_position, new_velocity, new_forces
