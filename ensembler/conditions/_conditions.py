"""
Module: Conditions
    This module shall be used to implement subclasses of conditions like, thermostat or distance restraints
"""

#ensembler.system import system as sys

class Condition:
    _tau:float  #tau = apply every tau steps
    nDim:int=0
    def __init__(self , sys):   #system):
        raise NotImplementedError("This " + __class__ + " class is not implemented")

    def apply(self):#, system:system):
        raise NotImplementedError("The function step in " + __class__ + " class is not implemented")

    def coupleSystem(self, system): #sys):
        self.system = system
        self.nDim = system.nDim
        self.nStates = system.nStates

class Constraint(Condition):
    pass

class Restraint(Condition):
    pass