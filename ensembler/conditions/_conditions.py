"""
Module: Conditions
    This module shall be used to implement subclasses of conditions like, thermostat or distance restraints
"""

from ensembler.util.basic_class import super_baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import system as systemType


class _conditionCls(super_baseClass):
    _tau: float  # tau = apply every tau steps
    nDim: int = 0

    def __init__(self, sys: systemType):
        notImplementedERR()

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
