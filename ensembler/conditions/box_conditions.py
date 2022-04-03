import numpy as np

from ensembler.conditions._basicCondition import _conditionCls
from ensembler.util.ensemblerTypes import systemCls as systemType, Iterable, Number, Union


class _boundaryCondition(_conditionCls):
    """
    This parent class is defining some functions for the actual box conditions.
    """

    lowerbounds: Iterable[Number]
    higherbounds: Iterable[Number]

    def __str__(self) -> str:
        msg = self.name + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
        msg += "\n"
        msg += "\tapply every step: " + str(self.nDim) + "\n"
        msg += "\tHigher bounds: " + str(self.higherbounds) + "\n"
        msg += "\tLower bounds: " + str(self.lowerbounds) + "\n"
        return msg

    def _parse_boundary(self, boundary: Union[Number, Iterable[Number]]) -> bool:
        if isinstance(boundary, Iterable):
            if all([isinstance(x, Number) for x in boundary]):
                self.higherbounds = np.array(np.max(boundary), ndmin=1)
                self.lowerbounds = np.array(np.min(boundary), ndmin=1)
            elif all([isinstance(x, Iterable) and [isinstance(y, Number) for y in x] for x in boundary]):
                self.higherbounds = np.max(boundary, axis=1)
                self.lowerbounds = np.min(boundary, axis=1)
            else:
                raise Exception("Could not read the Boundary Condition! : " + str(boundary))
            self.nDim = len(self.higherbounds)
        return True


class boxBoundaryCondition(_boundaryCondition):
    name: str = "Box Boundary Condition"

    def __init__(
        self,
        boundary: Union[Iterable[Number], Iterable[Iterable[Number]]],
        every_step: int = 1,
        system: systemType = None,
        verbose: bool = False,
    ):
        """box boundary condition
        This class can be used to define a box, which is restrictiring the phase space to its boundaries.
        Warning: This is not a very useful condition in most cases as the hard boundary leads to box effects in the simulation.
        We rather suggest the using the periodic boundary condition.
        Useage: Add this class as condition to a system.

        Parameters
        ----------
        boundary: Union[Iterable[Number], Iterable[Iterable[Number]]]
            defines the bounds of the system (in 3D-MD = box). Give here for each dimension the extrem coordinates.
        system: systemType, optional
            here a system can be given, normally this argument is not needed, as the system is coupling itself, if the condition is passed on construction to the system.
        """

        super().__init__(system=system, tau=every_step, verbose=verbose)

        self._parse_boundary(boundary=boundary)

    def apply(
        self, current_position: Union[Iterable[Number], Number], current_velocity: Union[Iterable[Number], Number]
    ) -> (Union[Iterable[Number], Number], Union[Iterable[Number], Number]):
        if self.verbose:
            print("box boundary_condition: before: ", current_position)
        current_position = np.array(current_position, ndmin=1)
        if self.verbose:
            print("CurrPos: ", current_position)
        for dim in range(len(current_position)):
            if self.verbose:
                print("DIM: ", dim)
            if current_position[dim] < self.lowerbounds[dim]:
                diff = abs(current_position[dim] - self.lowerbounds[dim])
                if self.verbose:
                    print("Lower:", diff, "Vel: ", current_velocity)
                current_position[dim] = self.lowerbounds[dim] + diff
                if not isinstance(current_velocity, type(None)):
                    current_velocity[dim] = np.nan if (current_velocity == np.nan) else -current_velocity[dim]
            elif current_position[dim] > self.higherbounds[dim]:
                diff = abs(current_position[dim] - self.higherbounds[dim])
                if self.verbose:
                    print("Higher:", diff)
                current_position[dim] = self.higherbounds[dim] - diff
                if not isinstance(current_velocity, type(None)):
                    current_velocity = np.nan if (current_velocity == np.nan) else -current_velocity[dim]

        if self.verbose:
            print("box boundary_condition: after: ", current_position)
        return np.squeeze(current_position), np.squeeze(current_velocity)

    def apply_coupled(self):
        """
        Applies the box Condition to the coupled system.
        """
        if self.system.step % self._tau == 0:
            newCurrentPosition, newCurrentVelocity = self.apply(
                current_position=self.system._currentPosition, current_velocity=self.system._currentVelocities
            )
            self.system._currentPosition = np.squeeze(newCurrentPosition)
            self.system._currentVelocity = np.squeeze(newCurrentVelocity)


class periodicBoundaryCondition(_boundaryCondition):
    name: str = "Periodic Boundary Condition"
    verbose: bool = False

    def __init__(
        self,
        boundary: Union[Iterable[Number], Iterable[Iterable[Number]]],
        every_step: int = 1,
        system: systemType = None,
        verbose: bool = False,
    ):
        """periodic boundary condition
           This class allows to enable sampling in mirror images and projects the coordinates to the restricted space.
           Add this class as condition to a system.

           Useage: Add this class as condition to a system.

        Parameters
        ----------
        boundary: Union[Iterable[Number], Iterable[Iterable[Number]]]
            defines the bounds of the system (in 3D-MD = box). Give here for each dimension the extrem coordinates.
        system: systemType, optional
            here a system can be given, normally this argument is not needed, as the system is coupling itself, if the condition is passed on construction to the system.
        """
        super().__init__(system=system, tau=every_step, verbose=verbose)
        self._parse_boundary(boundary)

    def apply(self, current_position: Union[Iterable[Number], Number]) -> Union[Iterable[Number], Number]:
        """
        apply the periodic boundary condition
        Parameters
        ----------
        current_position: Union[Iterable[Number], Number]
            current system position
        current_velocity: Union[Iterable[Number], Number]
            current systems velocity
        Returns
        -------
        Union[Iterable[Number], Number]
            new position
                a new position in the defined space
        """
        if self.verbose:
            print("periodic boundary_condition: before: ", current_position)
        current_position = np.array(current_position, ndmin=1)
        for dim in range(self.nDim):
            if current_position[dim] < self.lowerbounds[dim]:
                if self.verbose:
                    print("LOWER")
                current_position[dim] = self.higherbounds[dim] - (self.lowerbounds[dim] - current_position[dim])

            elif current_position[dim] > self.higherbounds[dim]:
                if self.verbose:
                    print("UPPER")
                current_position[dim] = self.lowerbounds[dim] + (current_position[dim] - self.higherbounds[dim])
            if self.verbose:
                print("periodic boundary_condition: after: ", current_position)

        return current_position

    def apply_coupled(self):
        """
        Applies the box Condition to the coupled system.
        """
        if self.system.step % self._tau == 0:
            newCurrentPosition = self.apply(current_position=self.system._currentPosition)
            self.system._currentPosition = np.squeeze(newCurrentPosition)
