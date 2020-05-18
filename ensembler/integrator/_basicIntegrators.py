"""
Module: integrator
    This module shall be used to implement subclasses of integrator. The integrators are use for propagating simulatoins.
    Think about how to link conditions to integrator???
"""


import numpy as np
from typing import Tuple
import scipy.constants as const

#from ensembler import system

class _integratorCls:
    """
    autoclass: integrator
        This class is the parent class to all other classes.
    """
    #general:
    verbose:bool = False
    nDim:int = 0

    #Params
    maxStepSize:float = None
    minStepSize:float = None
    spaceRange:tuple = None

    #calculation
    posShift:float = 0

    #Limits:
    _critInSpaceRange = lambda self,pos: self.spaceRange == None or (self.spaceRange != None and pos >= min(self.spaceRange) and pos <= max(self.spaceRange))

    def __init__(self):
        raise NotImplementedError("This "+str(__class__)+" class is not implemented")
    
    def step(self, system):
        """
        ..autofunction: step
            This is the parent function that is the interface for all integrator step functions.

        :param system: This is a system, that should be integrated.
        :type system: ensembler.system.system
        :return: (new Position, new velocity, position Shift/ force)
        :rtype: (float, float, float)
        """
        raise NotImplementedError("The function step in "+__class__+" class is not implemented")

    def integrate(self, system, steps:int):
        for step in range(steps):
            (newPosition, newVelocity, newForces) = self.step(system=system)
            system.append_state(newPosition=newPosition, newVelocity=newVelocity, newForces=newForces)

    def setVerbose(self, verbose:bool=True):
        self.verbose = verbose
