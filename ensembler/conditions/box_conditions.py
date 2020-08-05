import numpy as np

from ensembler.conditions._conditions import _conditionCls
from ensembler.util.ensemblerTypes import system as systemType, Iterable, Number, Union


class _boundaryCondition(_conditionCls):
    """
    This parent class is defining some functions for the actual conditions.
    """
    lowerbounds: Iterable[Number]
    higherbounds: Iterable[Number]
    verbose: bool = False

    def __str__(self) -> str:
        msg = self.name + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
        msg += "\n"
        msg += "\tapply every step: " + str(self.nDim) + "\n"
        msg += "\tHigher bounds: " + str(self.higherbounds) + "\n"
        msg += "\tLower bounds: " + str(self.lowerbounds) + "\n"
        return msg

    def _parse_boundary(self, boundary: Union[Number, Iterable[Number]]) -> bool:
        if (isinstance(boundary, Iterable)):
            if (all([isinstance(x, Number) for x in boundary])):
                self.higherbounds = np.array(np.max(boundary), ndmin=1)
                self.lowerbounds = np.array(np.min(boundary), ndmin=1)
            elif (all([isinstance(x, Iterable) and [isinstance(y, Number) for y in x] for x in boundary])):
                self.higherbounds = np.max(boundary, axis=1)
                self.lowerbounds = np.min(boundary, axis=1)
            else:
                raise Exception("Could not read the Boundary Condition! : " + str(boundary))
            self.nDim = len(self.higherbounds)
        return True


class boxBoundaryCondition(_boundaryCondition):
    name: str = "Box Boundary Condition"

    def __init__(self, boundary: Union[Iterable[Number], Iterable[Iterable[Number]]], every_step: int = 1,
                 system: systemType = None):
        """        box boundary condition
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
        self._parse_boundary(boundary=boundary)
        self._tau = every_step
        if (not isinstance(system, type(None))):
            self.system = system
            self.nDim = system.nDim
            self.nStates = system.nStates

    def apply(self, currentPosition: Union[Iterable[Number], Number],
              currentVelocity: Union[Iterable[Number], Number]) -> (
    Union[Iterable[Number], Number], Union[Iterable[Number], Number]):
        if self.verbose: print("box boundary_condition: before: ", currentPosition)
        currentPosition = np.array(currentPosition, ndmin=1)
        if self.verbose: print("CurrPos: ", currentPosition)
        for dim in range(len(currentPosition)):
            if self.verbose: print("DIM: ", dim)
            if (currentPosition[dim] < self.lowerbounds[dim]):
                diff = abs(currentPosition[dim] - self.lowerbounds[dim])
                if self.verbose: print("Lower:", diff, "Vel: ", currentVelocity)
                currentPosition[dim] = (self.lowerbounds[dim] + diff)
                currentVelocity[dim] = np.nan if (currentVelocity == np.nan) else -currentVelocity[dim]
            elif (currentPosition[dim] > self.higherbounds[dim]):
                diff = abs(currentPosition[dim] - self.higherbounds[dim])
                if self.verbose: print("Higher:", diff)
                currentPosition[dim] = (self.higherbounds[dim] - diff)
                currentVelocity = np.nan if (currentVelocity == np.nan) else -currentVelocity[dim]

        if self.verbose: print("box boundary_condition: after: ", currentPosition)
        return np.squeeze(currentPosition), np.squeeze(currentVelocity)

    def apply_coupled(self):
        """
        Applies the box Condition to the coupled system.
        """
        if (self.system.step % self._tau == 0):
            newCurrentPosition, newCurrentVelocity = self.apply(currentPosition=self.system._currentPosition,
                                                                currentVelocity=self.system._currentVelocities)
            self.system._currentPosition = np.squeeze(newCurrentPosition)
            self.system._currentVelocity = np.squeeze(newCurrentVelocity)


class periodicBoundaryCondition(_boundaryCondition):
    name: str = "Periodic Boundary Condition"
    verbose: bool = True

    def __init__(self, boundary: Union[Iterable[Number], Iterable[Iterable[Number]]], every_step: int = 1,
                 system: systemType = None):
        """        periodic boundary condition
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
        self._parse_boundary(boundary)
        self._tau = every_step

        if (system != None):
            self.system = system
            self.nDim = system.nDim
            self.nStates = system.nStates

    def apply(self, currentPosition: Union[Iterable[Number], Number],
              currentVelocity: Union[Iterable[Number], Number]) -> Union[Iterable[Number], Number]:
        if self.verbose: print("periodic boundary_condition: before: ", currentPosition)
        currentPosition = np.array(currentPosition, ndmin=1)
        for dim in range(self.nDim):
            if (currentPosition[dim] < self.lowerbounds[dim]):
                if self.verbose: print("LOWER")
                currentPosition[dim] = self.higherbounds[dim] - (self.lowerbounds[dim] - currentPosition[dim])
                currentVelocity[dim] *= -1
            elif (currentPosition[dim] > self.higherbounds[dim]):
                if self.verbose: print("UPPER")
                currentPosition[dim] = self.lowerbounds[dim] + (currentPosition[dim] - self.higherbounds[dim])
                currentVelocity[dim] *= -1
        if self.verbose: print("periodic boundary_condition: after: ", currentPosition)

        return currentPosition, currentVelocity

    def apply_coupled(self):
        """
        Applies the box Condition to the coupled system.
        """
        if (self.system.step % self._tau == 0):
            newCurrentPosition, newCurrentVelocity = self.apply(currentPosition=self.system._currentPosition,
                                                                currentVelocity=self.system._currentVelocities)
            self.system._currentPosition = np.squeeze(newCurrentPosition)
            self.system._currentVelocity = np.squeeze(newCurrentVelocity)
