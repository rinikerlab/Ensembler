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

'''
Langevin stochastic integration
'''

class langevinIntegrator(_integratorCls):


    def __init__(self, dt:float=0.005, gamma:float=50, oldPosition:float=None):
        self.dt = dt
        self.gamma = gamma
        self._oldPosition = oldPosition
        self._first_step = True # only neede for velocity Langevin
        self.R_x = None
        self.newForces = None
        self.currentPosition = None
        self.currentVelocity = None


    def update_positon(self, system):
        """
        Integrate step according to Position Langevin BBK integrator
        Designed after:
        Designed after: http://localscf.com/localscf.com/LangevinDynamics.aspx.html

        update position
            This interface function needs to be implemented for a subclass.
            The purpose of this function is to perform one integration step.

        Parameters
        ----------
        system : systemType
           A system, that should be integrated.

        Returns
        -------
        Tuple[float, None]
            This Tuple contains the new: (new Position, new velocity=None)
            for velocity return use langevinVelocityIntegrator

        Raises
        ------
        NotImplementedError
            You need to implement this function in the subclass (i.e. in your integrator)

        """

        nDim = system.nDim
        #get random number, normal distributed for nDim dimentions
        curr_random = np.squeeze(np.random.normal(0,1,nDim))
        #scale random number according to fluctuation-dissipation theorem
        # energy is expected to be in units of k_B
        self.R_x = np.sqrt(2 * system.temperature * self.gamma * system.mass / self.dt) * curr_random
        #calculation of forces:
        self.newForces = -system.potential.dvdpos(self.currentPosition)

        #Brünger-Brooks-Karplus integrator for positions
        new_position = (1 / (1 + self.gamma * self.dt/2)) * (2 * self.currentPosition - self._oldPosition
            + self.gamma * (self.dt / 2) * (self._oldPosition) + (self.dt**2 / system.mass) * (self.R_x + self.newForces))

        return new_position, None

    def step(self, system):
        """
        step
            This interface function needs to be implemented for a subclass.
            The purpose of this function is to perform one integration step.

        Parameters
        ----------
        system : systemType
           A system, that should be integrated.

        Returns
        -------
        Tuple[float, float, float]
            This Tuple contains the new: (new Position, new velocity, position Shift/ force)

        Raises
        ------
        NotImplementedError
            You need to implement this function in the subclass (i.e. in your integrator)

        """
        # get current positiona and velocity form system class
        self.currentPosition = system._currentPosition
        self.currentVelocity = system._currentVelocities

        # hirachy: first check if old postition is given, if not it takes the velocity from the system class
        # is there no initial velocity a Maxwell-Boltzmann distributied velocity is generated
        if self._oldPosition is None:
            # get old position from velocity, only during initialization
            print("initializing Langevin old Positions\t ")
            print("\n")
            self._oldPosition = self.currentPosition - self.currentVelocity * self.dt

        # integration step
        new_position, new_velocity = self.update_positon(system)
        # update position
        self._oldPosition = self.currentPosition



        if(self.verbose):
            print("INTEGRATOR: current forces\t ", self.newForces)
            print("INTEGRATOR: old Position\t ", sef._oldPosition)
            print("INTEGRATOR: current_position\t ", currentPosition)
            print("INTEGRATOR: current_velocity\t ", currentVelocity)
            print("INTEGRATOR: newPosition\t ", new_position)
            print("INTEGRATOR: newVelocity\t ", new_velocity)
            print("\n")
        return new_position, new_velocity, self.newForces  # add random number


class langevinVelocityIntegrator(langevinIntegrator):

    """
    Integrate step according to Velocity Langevin BKK integrator
    Designed after:
    Designed after: http://localscf.com/localscf.com/LangevinDynamics.aspx.html

    update position
        This interface function needs to be implemented for a subclass.
        The purpose of this function is to perform one integration step.

    Parameters
    ----------
    system : systemType
       A system, that should be integrated.

    Returns
    -------
    Tuple[float, None]
        This Tuple contains the new: (new Position, new velocity)

        returns both velocities and positions at full steps

    Raises
    ------
    NotImplementedError
        You need to implement this function in the subclass (i.e. in your integrator)

    """

    def update_positon(self, system):
        """
        update for position Lanvevin
        :return:
        """

        # for the first step we have to calculate new random numbers and forces
        # then we can take the one from  the previous  step
        nDim = system.nDim
        if self._first_step:
            # get random number, normal distributed for nDim dimentions
            curr_random = np.squeeze(np.random.normal(0, 1, nDim))
            # scale random number according to fluctuation-dissipation theorem
            # energy is expected to be in units of k_B
            self.R_x = np.sqrt(2 * system.temperature * self.gamma * system.mass / self.dt) * curr_random
            #calculate of forces:
            self.newForces = -system.potential.dvdpos(self.currentPosition)

            self._first_step = False

        #Brünger-Brooks-Karplus integrator for velocities

        half_step_velocity = (1-self.gamma*self.dt/2)*self.currentVelocity + self.dt/(2*system.mass) * (self.newForces + self.R_x)

        full_step_position = self.currentPosition + half_step_velocity*self.dt

        # calculate forces and random number for new position
        # get random number, normal distributed for nDim dimentions
        curr_random = np.squeeze(np.random.normal(0,1,nDim)) # for n dimentions
        # scale random number according to fluctuation-dissipation theorem
        # energy is expected to be in units of k_B
        self.R_x = np.sqrt(2 * system.temperature * self.gamma * system.mass / self.dt) * curr_random

        #calculate of forces:
        self.newForces = -system.potential.dvdpos(full_step_position)

        # last half step
        full_step_velocity = (1 / (1 + self.gamma * self.dt/2)) * (half_step_velocity +self.dt/(1*system.mass)*(self.newForces + self.R_x))

        return full_step_position, full_step_velocity