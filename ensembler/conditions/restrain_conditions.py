from ensembler.conditions._conditions import _conditionCls
from ensembler.potentials import OneD
from ensembler.util.ensemblerTypes import system as systemType, potential as potentialType, Iterable, Number, Union


class restraint(_conditionCls):
    pass


class positionRestraintCondition(restraint):
    name: str = "position restraint"
    position_0: Union[Number, Iterable[Number]]
    functional: potentialType

    def __init__(self, position_0: Union[Number, Iterable[Number]], every_step: int = 1,
                 restraint_functional: potentialType = OneD.harmonicOscillatorPotential, system: systemType = None):
        self.position_0 = position_0
        self.functional = restraint_functional(x_shift=position_0)
        self._tau = every_step

        if (not isinstance(system, type(None))):
            self.system = system
            self.nDim = system.nDim
            self.nStates = system.nStates
        else:
            self.nDim = self.functional.constants[self.functional.nDim]

    def __str__(self) -> str:
        msg = self.name + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
        msg += "\tReference Position: " + str(self.position_0) + "\n"
        msg += "\tapply every step: " + str(self._tau) + "\n"
        msg += "\tfunctional: " + str(self.functional.name) + "\n"
        return msg

    def apply(self, currentPosition: Union[Iterable[Number], Number]):
        return self.functional.ene(currentPosition), self.functional.force(currentPosition)

    def apply_coupled(self, ):
        if (self.system.step % self._tau == 0):
            newCurrentPosition, newCurrentVelocity = self.apply(currentPosition=self.system._currentPosition, )
            self.system._currentPosition += newCurrentPosition
            self.system._currentForce += newCurrentVelocity
