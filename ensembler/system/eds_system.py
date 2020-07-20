
import numpy as np
from typing import Iterable, NoReturn
from numbers import Number
import pandas as pd
import scipy.constants as const
pd.options.mode.use_inf_as_na = True

from ensembler.util import dataStructure as data

from ensembler.potentials import OneD as pot

from ensembler.util import  ensemblerTypes as ensemblerTypes
_integratorCls = ensemblerTypes.integrator
_conditionCls = ensemblerTypes.condition

from ensembler.system.basic_system import system
from ensembler.integrator.stochastic import metropolisMonteCarloIntegrator


class edsSystem(system):
    """
    
    """

    #Lambda Dependend Settings
    state = data.envelopedPStstate
    currentState: data.envelopedPStstate
    potential: pot.envelopedPotential

    #current lambda
    _currentEdsS:float = np.nan
    _currentEdsEoffs:float = np.nan


    def __init__(self, potential:pot.envelopedPotential=pot.envelopedPotential(V_is=[pot.harmonicOscillator(x_shift=2), pot.harmonicOscillator(x_shift=-2)], Eoff_i=[0,0]),
                 integrator: _integratorCls=metropolisMonteCarloIntegrator(), conditions: Iterable[_conditionCls]=[],
                 temperature: float = 298.0, position:(Iterable[Number] or float) = None, eds_s=1, eds_Eoff=[0, 0]):

        ################################
        # Declare Attributes
        #################################
        
        self.state = data.envelopedPStstate
        self._currentEdsS = eds_s
        self._currentEdsEoffs = eds_Eoff
        
        ##Physical parameters
        self.temperature: float = 298.0
        self.mass: float = 1  # for one particle systems!!!!
        self.nparticles: int = 1  # Todo: adapt it to be multiple particles

        self.nDim: int = -1
        self.nStates: int = 1

        # Output
        self.initial_position: Iterable[float] or float

        self.currentState: data.basicState = data.basicState(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        self.trajectory: pd.DataFrame = pd.DataFrame(columns=list(self.state.__dict__["_fields"]))

        # tmpvars - private:
        self._currentTotE: (Number) = np.nan
        self._currentTotPot: (Number) = np.nan
        self._currentTotKin: (Number) = np.nan
        self._currentPosition: (Number or Iterable[Number]) = np.nan
        self._currentVelocities: (Number or Iterable[Number]) = np.nan
        self._currentForce: (Number or Iterable[Number]) = np.nan
        self._currentTemperature: (Number or Iterable[Number]) = np.nan


        super().__init__(potential=potential, integrator=integrator, conditions=conditions, temperature=temperature, position=position)
        self.set_s(self._currentEdsS)
        self.set_Eoff(self._currentEdsEoffs)

    def set_current_state(self, currentPosition:(Number or Iterable), currentLambda:(Number or Iterable),
                          currentVelocities:(Number or Iterable)=0,  current_s:(Number or Iterable)=0, current_Eoff:(Number or Iterable)=0,
                          currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self._currentEdsS = current_s
        self._currentEdsEoffs = current_Eoff

        self.updateSystemProperties()
        self.updateCurrentState()

    def updateCurrentState(self):
        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=self._currentTotE,
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       s=self._currentEdsS, Eoff=self._currentEdsEoffs)

    def append_state(self, newPosition, newVelocity, newForces, newS, newEoff):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces
        self._currentEdsS = newS
        self._currentEdsEoffs = newEoff

        self.updateSystemProperties()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def set_s(self, s):
        self._currentEdsS = s
        self.potential.set_s(s=self._currentEdsS)
        self.updateSystemProperties()

    def set_Eoff(self, Eoff: Iterable[float]):
        self._currentEdsEoffs = Eoff
        self.potential.set_Eoff(Eoff=self._currentEdsEoffs)
        self.updateSystemProperties()
