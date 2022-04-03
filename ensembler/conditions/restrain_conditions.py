from ensembler.conditions._basicCondition import _conditionCls
from ensembler.potentials import OneD
from ensembler.util.ensemblerTypes import systemCls as systemType, potentialCls as potentialType, Iterable, Number, Union, Tuple, NoReturn


class Restraint(_conditionCls):
    pass


class positionRestraintCondition(Restraint):
    """
    The position restraint is adding a bias to a certain position selected in the coordinate space
    """

    name: str = "position restraint"
    position_0: Union[Number, Iterable[Number]]
    functional: potentialType

    def __init__(
        self,
        position_0: Union[Number, Iterable[Number]],
        every_step: int = 1,
        restraint_functional: potentialType = OneD.harmonicOscillatorPotential,
        system: systemType = None,
        verbose: bool = False,
    ):
        """
            __init__
                builds the bias to a certain position_0 in the coordinate space, which is applied every_step
        Parameters
        ----------
        position_0: Union[Number, Iterable[Number]]
            coordinate space position
        every_step: int, optional
            apply bias every_step. (default=1)
        restraint_functional: potentialType, optional
            potential function for the bias (default: harmonic oscillator)
        system: systemType, optional
            system to couple to (default: None -> not coupled)
        verbose: bool, optional
            shall I tell you a story?
        """
        self.position_0 = position_0
        self.functional = restraint_functional(x_shift=position_0)

        super().__init__(system=system, tau=every_step, verbose=verbose)

    def __str__(self) -> str:
        msg = self.name + "\n"
        msg += "\tDimensions: " + str(self.nDimensions) + "\n"
        msg += "\tReference Position: " + str(self.position_0) + "\n"
        msg += "\tapply every step: " + str(self._tau) + "\n"
        msg += "\tfunctional: " + str(self.functional.name) + "\n"
        return msg

    def apply(self, current_position: Union[Iterable[Number], Number]) -> Tuple[Number, Number]:
        """
            apply
                applies the condition

        Parameters
        ----------
        current_position: Union[Iterable[Number], Number]
            the current to position to bias

        Returns
        -------
        Tuple[Number, Number]
            the potential energy bias, the force bias.
        """
        return self.functional.ene(current_position), self.functional.force(current_position)

    def apply_coupled(self) -> NoReturn:
        """
        apply the condition to the coupled system.
        """
        if self.system.step % self._tau == 0:
            new_current_position, new_current_velocity = self.apply(current_position=self.system._currentPosition)
            self.system._currentPosition += new_current_position
            self.system._currentForce += new_current_velocity
