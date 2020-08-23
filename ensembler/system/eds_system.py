from numbers import Number
from typing import Iterable

import numpy as np
import pandas as pd

pd.options.mode.use_inf_as_na = True

from ensembler.util import dataStructure as data

from ensembler.potentials import OneD as pot

from ensembler.util import ensemblerTypes as ensemblerTypes

_integratorCls = ensemblerTypes.sampler
_conditionCls = ensemblerTypes.condition

from ensembler.system.basic_system import system
from ensembler.samplers.stochastic import metropolisMonteCarloIntegrator


class edsSystem(system):
    """
    
    """
    name = "eds system"
    # Lambda Dependend Settings
    state = data.envelopedPStstate
    currentState: data.envelopedPStstate
    potential: pot.envelopedPotential

    # current lambda
    _currentEdsS: float = np.nan
    _currentEdsEoffs: float = np.nan

    def __init__(self, potential: pot.envelopedPotential = pot.envelopedPotential(
        V_is=[pot.harmonicOscillatorPotential(x_shift=2), pot.harmonicOscillatorPotential(x_shift=-2)], Eoff_i=[0, 0]),
                 sampler: _integratorCls = metropolisMonteCarloIntegrator(),
                 conditions: Iterable[_conditionCls] = [],
                 temperature: float = 298.0, start_position: (Iterable[Number] or float) = None, eds_s=1, eds_Eoff=[0, 0]):
        ################################
        # Declare Attributes
        #################################

        self._currentEdsS = eds_s
        self._currentEdsEoffs = eds_Eoff
        self.state = data.envelopedPStstate

        super().__init__(potential=potential, sampler=sampler, conditions=conditions, temperature=temperature,
                         start_position=start_position)


        # Output
        self.set_s(self._currentEdsS)
        self.set_Eoff(self._currentEdsEoffs)

    def set_current_state(self, currentPosition: (Number or Iterable), currentLambda: (Number or Iterable),
                          currentVelocities: (Number or Iterable) = 0, current_s: (Number or Iterable) = 0,
                          current_Eoff: (Number or Iterable) = 0,
                          currentForce: (Number or Iterable) = 0, currentTemperature: Number = 298):
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

    @property
    def s(self):
        return self._currentEdsS

    @s.setter
    def s(self, s: Number):
        self._currentEdsS = s
        self.potential.set_s(self._currentEdsS)
        self.updateSystemProperties()

    def set_s(self, s: Number):
        self.s = s

    @property
    def Eoff(self):
        return self._currentEdsEoffs

    @Eoff.setter
    def Eoff(self, Eoff: Iterable[float]):
        self._currentEdsEoffs = Eoff
        self.potential.Eoff_i = self._currentEdsEoffs
        self.updateSystemProperties()

    def set_Eoff(self, Eoff: Iterable[float]):
        self.Eoff = Eoff
