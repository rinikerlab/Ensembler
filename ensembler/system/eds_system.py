
import numpy as np
from typing import Iterable, NoReturn
from numbers import Number
import pandas as pd
import scipy.constants as const
pd.options.mode.use_inf_as_na = True

from ensembler.system.basic_system import system
from ensembler.util import dataStructure as data

from ensembler import potentials
from ensembler.integrator import _integratorCls, metropolisMonteCarloIntegrator
from ensembler.conditions._conditions import Condition

class edsSystem(system):
    """
    
    """

    #Lambda Dependend Settings
    state = data.envelopedPStstate
    currentState: data.envelopedPStstate
    potential: potentials.ND.envelopedPotential

    #current lambda
    _current_eds_s:float = np.nan
    _current_eds_Eoff:float = np.nan


    def __init__(self, potential:potentials.ND.envelopedPotential=potentials.OneD.envelopedPotential(V_is=[potentials.OneD.harmonicOscillator(x_shift=2), potentials.OneD.harmonicOscillator(x_shift=-2)], Eoff_i=[0,0]), 
                 integrator: _integratorCls=metropolisMonteCarloIntegrator(), conditions: Iterable[Condition]=[],
                 temperature: float = 298.0, position:(Iterable[Number] or float) = None, lam:float=0.0, eds_s=1, eds_Eoff=[0, 0]):

        self.state = data.envelopedPStstate
        self._current_eds_s = eds_s
        self._current_eds_Eoff = eds_Eoff


        super().__init__(potential=potential, integrator=integrator, conditions=conditions, temperature=temperature, position=position)
        self.set_s(self._current_eds_s)
        self.set_Eoff(self._current_eds_Eoff)

    def set_current_state(self, currentPosition:(Number or Iterable), currentLambda:(Number or Iterable),
                          currentVelocities:(Number or Iterable)=0,  current_s:(Number or Iterable)=0, current_Eoff:(Number or Iterable)=0,
                          currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self._current_eds_s = current_s
        self._current_eds_Eoff = current_Eoff

        self.updateEne()
        self.updateCurrentState()

    def updateCurrentState(self):
        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=self._currentTotE,
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       s=self._current_eds_s, Eoff=self._current_eds_Eoff)

    def append_state(self, newPosition, newVelocity, newForces, newS, newEoff):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces
        self._currentEdsS = newS
        self._currentEdsEoffs = newEoff

        self.updateTemp()
        self.updateEne()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def set_s(self, s):
        self._currentEdsS = s
        self.potential.set_s(s=self._currentEdsS)
        self.updateEne()

    def set_Eoff(self, Eoff: Iterable[float]):
        self._currentEdsEoffs = Eoff
        self.potential.set_Eoff(Eoff=self._currentEdsEoffs)
        self.updateEne()
