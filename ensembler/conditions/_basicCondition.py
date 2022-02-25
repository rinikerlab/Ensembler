"""
Module: Conditions
    This module shall be used to implement subclasses of conditions like, thermostat or distance restraints
"""

from ensembler.util.basic_class import _baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import systemCls as systemType, NoReturn


class _conditionCls(_baseClass):
    """
    This class provides the basic functionality for the condition classes.
    """

    nDimensions: int = 0
    nStates: int = 0

    def __init__(self, system: systemType = None, tau: int = 1, verbose: bool = False):
        """
            __init__
                set the basic variables for the condition class.

        Parameters
        ----------
        system: systemType, optional
            is the system, to couple the condition to. (default: None - we recommend providing a system)
        tau: int, optional
            apply condition each tau step.(default: 1)
        verbose: bool, optional
            noise! (default: False)
        """

        self._tau = tau
        self._verbose = verbose
        self._system = system

        if system != None:
            self.nDim = system.nDim
            self.nStates = system.nStates

        if system is not None:
            self.couple_system(system=system)

    @property
    def tau(self) -> int:
        """
            apply the condition every tau step.
        Returns
        -------
        int
            each tau steps
        """
        return self._tau

    @tau.setter
    def tau(self, tau: int):
        self._tau = tau

    @property
    def system(self) -> systemType:
        """
            the system, the condition is applied to.

        Returns
        -------
        systemType
            coupled system
        """
        return self._system

    @system.setter
    def system(self, system: systemType):
        self._system = system

    @staticmethod
    def apply(self):
        """
        apply function interface. Needs to be implemented. - takes all needed parameters
        """
        notImplementedERR()

    def apply_coupled(self) -> NoReturn:
        """
        apply_coupled function interface. Needs to be implemented if you want to use the condition with a system.
        takes the coupled system and passes the correct parameters to apply.
        """
        notImplementedERR()

    def couple_system(self, system: systemType) -> NoReturn:
        """
            Couple the given system to the condition.

        Parameters
        ----------
        system: systemType
            the system to be coupled

        Returns
        -------

        """
        self.system = system
        self.nDimensions = system.nDimensions
        self.nStates = system.nStates
