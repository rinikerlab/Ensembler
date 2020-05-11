
import numpy as np
from typing import Iterable, NoReturn
from numbers import Number
import pandas as pd
import scipy.constants as const
pd.options.mode.use_inf_as_na = True


from ensembler.util import dataStructure as data
from ensembler.potentials.ND import envelopedPotential
from ensembler.potentials._baseclasses import _potentialNDCls as _potentialCls
from ensembler.potentials._baseclasses import _potential1DClsSymPYPerturbed as _perturbedPotentialCls

from ensembler.system.basic_system import system
from ensembler.integrator import _integratorCls
from ensembler.conditions._conditions import Condition

class perturbedSystem(system):
    """
    
    """

    #Lambda Dependend Settings
    state = data.lambdaState
    currentState: data.lambdaState
    potential: _perturbedPotentialCls

    #current lambda
    _currentLam:float = np.nan
    _currentdHdLam:float = np.nan


    def __init__(self, potential:_perturbedPotentialCls, integrator: _integratorCls, conditions: Iterable[Condition]=[],
                 temperature: float = 298.0, position:(Iterable[Number] or float) = None, lam:float=0.0):
        self._currentLam = lam
        super().__init__(potential=potential, integrator=integrator, conditions=conditions, temperature=temperature, position=position)
        self.set_lam(lam)

    def set_current_state(self, currentPosition:(Number or Iterable), currentLambda:(Number or Iterable),
                          currentVelocities:(Number or Iterable)=0,  currentdHdLam:(Number or Iterable)=0,
                          currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self.updateEne()
        self.updateCurrentState()

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

        self.updateTemp()
        self.updateEne()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def set_lam(self, lam:float):
        self._currentLam = lam
        self.potential.set_lam(lam=self._currentLam)
        self.updateEne()

    def _update_dHdlambda(self):
        self._currentdHdLam = self.potential.dhdlam(self._currentPosition)
        self.updateCurrentState()
        return self._currentdHdLam