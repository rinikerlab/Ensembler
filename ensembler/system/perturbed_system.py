from numbers import Number
from typing import Iterable, NoReturn

import numpy as np
import pandas as pd

pd.options.mode.use_inf_as_na = True

from ensembler.util import dataStructure as data

from ensembler.util import ensemblerTypes as ensemblerTypes

_integratorCls = ensemblerTypes.sampler
_conditionCls = ensemblerTypes.condition

from ensembler.potentials._basicPotentials import _potential1DClsPerturbed as _perturbedPotentialCls
from ensembler.system.basic_system import system


class perturbedSystem(system):
    """
    
    """
    name = "perturbed system"
    # Lambda Dependend Settings
    state = data.lambdaState
    currentState: data.lambdaState
    potential: _perturbedPotentialCls

    # current lambda
    _currentLam: float = np.nan
    _currentdHdLam: float = np.nan

    def __init__(self, potential: _perturbedPotentialCls, sampler: _integratorCls,
                 conditions: Iterable[_conditionCls] = [],
                 temperature: float = 298.0, start_position: (Iterable[Number] or float) = None, lam: float = 0.0):
        self._currentLam = lam
        super().__init__(potential=potential, sampler=sampler, conditions=conditions, temperature=temperature,
                         start_position=start_position)
        self.set_lam(lam)

    def set_current_state(self, currentPosition: (Number or Iterable), currentLambda: (Number or Iterable),
                          currentVelocities: (Number or Iterable) = 0, currentdHdLam: (Number or Iterable) = 0,
                          currentForce: (Number or Iterable) = 0, currentTemperature: Number = 298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self._updateEne()
        self._update_dHdlambda()
        self.updateCurrentState()

    def updateSystemProperties(self) -> NoReturn:
        self._updateEne()
        self._updateTemp()
        self._update_dHdlambda()

    def updateCurrentState(self):
        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=self._currentTotE,
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       lam=self._currentLam, dhdlam=self._currentdHdLam)

    def append_state(self, newPosition, newVelocity, newForces, newLam):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces
        self._currentLam = newLam

        self._updateTemp()
        self._updateEne()
        self._update_dHdlambda()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def set_lam(self, lam: float):
        self._currentLam = lam
        self.potential.set_lam(lam=self._currentLam)
        self._updateEne()

    def _update_dHdlambda(self):
        self._currentdHdLam = self.potential.dvdlam(self._currentPosition)
        self.updateCurrentState()
        return self._currentdHdLam
