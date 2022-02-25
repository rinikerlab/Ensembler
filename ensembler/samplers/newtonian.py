"""
Module: Sampler
    The sampler module is provides methods exploring the potential functions.

    Newtonian Integrators

"""

from ensembler.samplers._basicSamplers import _samplerCls
from ensembler.util.ensemblerTypes import systemCls as systemType, Union, Number


class newtonianSampler(_samplerCls):
    """
    This class is the parent class for all newtonian samplers. The pre-implemented
    newtonian type samplers currently comprise the Velocity Verlet, Position Verlet and
    Leapfrog integrator.
    """

    current_position: Number
    current_velocity: Number
    current_forces: Number

    dt: float

    def __init__(self, dt=0.002):
        """
        __init__
            This is the Constructor of the newtonian samplers.

        Parameters
        ----------
        dt: Number, optional
            time step of an integration, by default 0.002
        """
        super().__init__()
        self.dt = dt


class velocityVerletIntegrator(newtonianSampler):
    """
    The velocity Verlet Integrator is one of the simplest integrators that provides good numerical stability as well
    as time-reversibility and symplectic properties.
    It's local error in position is of order dt^4 and the local error in verlocity is of order dt^2.

    Verlet, Loup (1967). "Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard−Jones Molecules". Physical Review. 159 (1): 98–103.
    """

    name = "Verlocity Verlet Integrator"

    def step(self, system: systemType) -> Union[Number, Number, Number]:
        """
        step
            This function is performing an integration step in Verlocity Verlet fashion.

        Parameters
        ----------
        system : systemType
            A system, that should be integrated.

        Returns
        -------
        Tuple[Number, Number, Number]
            This Tuple contains the new: (new Position, new Velocity, new Force)
        """
        # init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities
        currentForces = system._currentForce

        # calculation:
        new_position = currentPosition + currentVelocity * self.dt - ((0.5 * currentForces * (self.dt**2)) / system.mass)
        new_forces = system.potential.force(new_position)
        new_velocity = currentVelocity - ((0.5 * (currentForces + new_forces) * self.dt) / system.mass)

        if self.verbose:
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")

        return new_position, new_velocity, new_forces


class positionVerletIntegrator(newtonianSampler):
    """
    The position Verlet Integrator has similar properties as the verlocity Verlet Integrator.

    Verlet, Loup (1967). "Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard−Jones Molecules". Physical Review. 159 (1): 98–103.
    """

    name = "Position Verlet Integrator"

    def step(self, system: systemType) -> Union[Number, Number, Number]:
        """
        step
            This function is performing an integration step in Position Verlet fashion.

        Parameters
        ----------
        system : systemType
            A system, that should be integrated.

        Returns
        -------
        Tuple[Number, Number, Number]
            This Tuple contains the new: (new Position, new Velocity, new Force)
        """
        # init
        currentPosition = system._currentPosition
        currentVelocity = system._currentVelocities

        # calculation:
        new_forces = system.potential.force(currentPosition)
        new_velocity = currentVelocity - (new_forces * self.dt / system.mass)
        new_position = currentPosition + new_velocity * self.dt

        if self.verbose:
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")
        return new_position, new_velocity, new_forces


class leapFrogIntegrator(newtonianSampler):
    """
    The leapFrogIntegrator is similar to the velocity Verlet method. Leapfrog integration is equivalent to
    updating positions and velocities at interleaved time points, staggered in such a way that they "leapfrog"
    over each other.
    """

    name = "Leap Frog Integrator"

    def step(self, system: systemType) -> Union[Number, Number, Number]:
        """
        step
            This function is performing an integration step in leapFrog fashion.

        Parameters
        ----------
        system : systemType
            A system, that should be integrated.

        Returns
        -------
        Tuple[Number, Number, Number]
            This Tuple contains the new: (new Position, new Velocity, new Force)
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

        if self.verbose:
            print(str(self.__name__) + ": current forces\t ", new_forces)
            print(str(self.__name__) + ": current Velocities\t ", currentVelocity)
            print(str(self.__name__) + ": current_position\t ", currentPosition)
            print(str(self.__name__) + ": newVel\t ", new_velocity)
            print(str(self.__name__) + ": newPosition\t ", new_position)
            print("\n")

        return new_position, new_velocity, new_forces
