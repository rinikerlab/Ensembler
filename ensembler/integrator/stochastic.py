"""
 Stochastic Integrators

"""

import numpy as np
from typing import Tuple
import scipy.constants as const

from ensembler import system
from ensembler.integrator._basicIntegrators import _integratorCls


class monteCarloIntegrator(_integratorCls):
    """
    ..autoclass: monteCarloIntegrator
        This class implements the classic monte carlo integrator.
        It choses its moves purely randomly.
    """
    resolution:float = 0.01   #increase the ammount of different possible values = between 0 and 10 there are 10/0.01 different positions.
    fixedStepSize: (float or list)

    def __init__(self, maxStepSize:float=None, minStepSize:float=None, spaceRange:tuple=None, fixedStepSize=None):
        self.fixedStepSize =  None if(isinstance(fixedStepSize, type(None))) else np.array(fixedStepSize)
        self.maxStepSize = maxStepSize
        self.minStepSize = minStepSize
        self.spaceRange = spaceRange
        pass
    
    def step(self, system)-> Tuple[float, None, float]:
        """
        ..autofunction: step
            This function is performing an integration step in MonteCarlo fashion.
        :param system: This is a system, that should be integrated.
        :type system: ensembler.system.system
        :return: (new Position, None, position Shift)
        :rtype: (float, None, float)
        """
        # integrate
        # while no value in spaceRange was found, terminates in first run if no spaceRange
        while(True):
            current_state = system.currentState

            self.oldpos = current_state.position
            self.randomShift(system.nDim)
            self.newPos = np.add(self.oldpos,self.posShift)

            #only get positions in certain range or accept if no range
            if(self._critInSpaceRange(self.newPos)):
                break
            else:
                self.newPos = self.oldpos           #reject step outside of range

        return self.newPos, np.nan, self.posShift
    
    def randomShift(self, nDim:int)->float:
        """
        ..autofunction: randomShift
            This function calculates the shift for the current position.

        :return: position shift
        :rtype: float
        """
        #which sign will the shift have?
        sign = np.array([-1 if(x <50) else 1 for x in np.random.randint(low=0, high=100, size=nDim)])

        #Check if there is a space restriction? - converges faster
        ##TODO: Implement functional
        if(not isinstance(self.fixedStepSize, type(None))):
            shift = self.fixedStepSize
        elif(self.spaceRange!=None):
            shift = np.multiply(np.abs(np.random.randint(low=self.spaceRange[0]/self.resolution, high=self.spaceRange[1]/self.resolution, size=nDim)), self.resolution)
        else:
            shift = np.abs(np.random.rand(nDim))
        #print(sign, shift)
        #Is the step shift in the allowed area? #Todo: fix min and max for mutliDimensional
        if(self.maxStepSize != None and shift > self.maxStepSize):#is there a maximal step size?
            self.posShift = np.multiply(sign, self.maxStepSize)
        elif(self.minStepSize != None and shift < self.minStepSize):
            self.posShift = np.multiply(sign, self.minStepSize)
        else:
            self.posShift = np.multiply(sign, shift)

        if(nDim == 1):  #TODO Make Effiecient?
            self.posShift = self.posShift[0]

        return self.posShift


class metropolisMonteCarloIntegrator(monteCarloIntegrator):
    """
    ..autoclass: metropolisMonteCarloInegrator
        This class is implementing a metropolis monte carlo Integrator.
        In opposite to the Monte Carlo Integrator, that is completley random, this integrator has limitations to the randomness.
        Theis limitation is expressed in the Metropolis Criterion.

        There is a standard Metropolis Criterion implemented, but it can also be exchanged with a different one.

        Default Metropolis Criterion:
            $ decision =  (E_{t} < E_{t-1}) ||  ( rand <= e^{(-1/(R/T*1000))*(E_t-E_{t-1})}$
            with:
                - $R$ as universal gas constant

        The original Metropolis Criterion (Nicholas Metropolis et al.; J. Chem. Phys.; 1953 ;doi: https://doi.org/10.1063/1.1699114):

            $ p_A(E_{t}, E_{t-1}, T) = min(1, e^{-1/(k_b*T) * (E_{t} - E_{t-1})})
            $ decision:  True if( 0.5 < p_A(E_{t}, E_{t-1}, T)) else False
            with:
                - $k_b$ as Boltzmann Constant
    """
    #
    #Parameters:
    metropolisCriterion=None    #use a different Criterion
    randomnessIncreaseFactor:float = 1  #tune randomness of your results
    maxIterationTillAccept:float = 100  #how often shall the integrator iterate till it accepts a step forcefully

    #METROPOLIS CRITERION
    ##random part of Metropolis Criterion:
    _defaultRandomness = lambda self, ene_new, currentState: ((1/self.randomnessIncreaseFactor)*np.random.rand() <= np.exp(-1.0 / (const.gas_constant / 1000.0 * currentState.temperature) * (ene_new - currentState.totPotEnergy))) #pseudocount  for equal energies
    ##default Metropolis Criterion
    _defaultMetropolisCriterion = lambda self, ene_new, currentState: (ene_new < currentState.totEnergy or self._defaultRandomness(ene_new, currentState))
    ## original criterion not useful causes overflows:
    #_defaultMetropolisCriterion = lambda self, ene_new, currentState: True if(0.5 > min(1, np.e**(-1/(const.k * currentState.temperature)*(ene_new-currentState.totPotEnergy)))) else False

    def __init__(self, minStepSize:float=None, maxStepSize:float=None, spaceRange:tuple=None, metropolisCriterion=None, randomnessIncreaseFactor=1, maxIterationTillAccept:int=100, fixedStepSize=None):
        self.fixedStepSize = None if(isinstance(fixedStepSize, type(None))) else np.array(fixedStepSize)
        self.maxStepSize = maxStepSize
        self.minStepSize = minStepSize
        self.spaceRange = spaceRange
        self.randomnessIncreaseFactor = randomnessIncreaseFactor
        self.maxIterationTillAccept = maxIterationTillAccept
        if(metropolisCriterion == None):
            self.metropolisCriterion = self._defaultMetropolisCriterion
        else:
            self.metropolisCriterion = metropolisCriterion

    def step(self, system):
        """
        ..autofunction: step
            This function is performing an integration step in MetropolisMonteCarlo fashion.
        :param system: This is a system, that should be integrated.
        :type system: ensembler.system.system
        :return: (new Position, None, position Shift)
        :rtype: (float, None, float)
        """

        iterstep = 0
        current_state = system.currentState
        self.oldpos = current_state.position
        nDim = system.nDim
        # integrate position
        while(True):    #while no value in spaceRange was found, terminates in first run if no spaceRange
            self.randomShift(nDim)
            #eval new Energy
            system._currentPosition = np.add(self.oldpos, self.posShift)
            ene = system.totPot()

            #MetropolisCriterion
            if ((self._critInSpaceRange(system._currentPosition) and self.metropolisCriterion(ene, current_state)) or iterstep==self.maxIterationTillAccept):
                break
            else:   #not accepted
                iterstep += 1
                continue
        return system._currentPosition , None, self.posShift
