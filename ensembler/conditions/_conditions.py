"""
Module: Conditions
    This module shall be used to implement subclasses of conditions like, thermostat or distance restraints
"""

from ensembler.util.basic_class import super_baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import system as systemType


class _conditionCls(super_baseClass):
    nDim: int = 0

    def __init__(self, system: systemType, tau:int, verbose:bool=False):
        self._tau = tau
        self._verbose = verbose
        self._system = system

    @property
    def tau(self)->int:
        return self._tau

    @tau.setter
    def tau(self, tau:int):
        self._tau = tau

    @property
    def system(self)->systemType:
        return self._system

    @system.setter
    def system(self, system: systemType):
        self._system = system

    def apply(self):
        notImplementedERR()

    def apply_coupled(self):
        notImplementedERR()

    def coupleSystem(self, system):
        self.system = system
        self.nDim = system.nDim
        self.nStates = system.nStates


class Constraint(_conditionCls):
    pass


class Restraint(_conditionCls):
    pass
