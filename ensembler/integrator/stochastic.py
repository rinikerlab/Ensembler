"""
 Stochastic Integrators

"""

import numpy as np
import scipy.constants as const

from ensembler.util.ensemblerTypes import system as systemType
from ensembler.util.ensemblerTypes import Union, List, Tuple, Number, Callable
from ensembler.integrator._basicIntegrators import _integratorCls

class stochasticIntegrator(_integratorCls):
    #Params
    minStepSize:Number=None
    maxStepSize:Number=1

    spaceRange:Tuple[Number, Number] = None
    resolution:float = 0.01   #increase the ammount of different possible values = between 0 and 10 there are 10/0.01 different positions. only used with space_range

    fixedStepSize: (Number or List[Number])

    #calculation
    posShift:float = 0  
    
    #Limits:
    _critInSpaceRange = lambda self,pos: self.spaceRange == None or (self.spaceRange != None and pos >= min(self.spaceRange) and pos <= max(self.spaceRange))

    def randomShift(self, nDim:int)->Union[float, np.array]:
        """
        randomShift 
            This function calculates the shift for the current position.

        Parameters
        ----------
        nDim : int
            gives the dimensionality of the position, defining the ammount of shifts.

        Returns
        -------
        Union[float, List[float]]
            returns the Shifts
        """

        #which sign will the shift have?
        sign = np.array([-1 if(x <50) else 1 for x in np.random.randint(low=0, high=100, size=nDim)])

        #Check if there is a space restriction? - converges faster
        if(not isinstance(self.fixedStepSize, type(None))):
            shift = np.array(np.full(shape=nDim, fill_value=self.fixedStepSize), ndmin=1)
        elif(not isinstance(self.spaceRange, type(None))):
            shift = np.array(np.multiply(np.abs(np.random.randint(low=np.min(self.spaceRange)/self.resolution, high=np.max(self.spaceRange)/self.resolution, size=nDim)), self.resolution), ndmin=1)
        else:
            shift = np.array(np.abs(np.random.rand(nDim)), ndmin=1)*self.maxStepSize

        self.posShift = np.multiply(sign, shift)

        
        #Is the step shift in the allowed area? #Todo: fix min and max for mutliDimensional
        if(self.minStepSize != None and any([s < self.minStepSize for s in shift])):
            self.posShift = np.multiply(sign, np.array([s if(s>self.minStepSize) else self.minStepSize for s in shift]) )
        else:
            self.posShift = np.multiply(sign, shift)
        

        return np.squeeze(self.posShift)

    

class monteCarloIntegrator(stochasticIntegrator):
    """
    monteCarloIntegrator 
        This class implements the classic monte carlo integrator.
        It choses its moves purely randomly.
    """

    def __init__(self, maxStepSize:Number=1, minStepSize:Number=None, spaceRange:Tuple[Number,Number]=None, fixedStepSize:Number=None):
        """
        __init__ 
            This is the Constructor of the MonteCarlo integrator.

        Parameters
        ----------
        maxStepSize : Number, optional
            maximal size of an integrationstep in any direction, by default 1
        minStepSize : Number, optional
            minimal size of an integration step in any direction, by default None
        spaceRange : Tuple[Number, Number], optional
            maximal and minimal allowed position for after an integration step. 
            If not fullfilled, step is rejected. By default None
        fixedStepSize : Number, optional
            this option restrains each integration step to a certain size in each dimension, by default None
        """
        self.fixedStepSize =  None if(isinstance(fixedStepSize, type(None))) else np.array(fixedStepSize)
        self.maxStepSize = maxStepSize
        self.minStepSize = minStepSize
        self.spaceRange = spaceRange
        pass
    
    def step(self, system:systemType)-> Tuple[float, None, float]:
        """
        step 
            This function is performing an integration step in MonteCarlo fashion.

        Parameters
        ----------
        system : systemType
           A system, that should be integrated.

        Returns
        -------
        Tuple[float, None, float]
            This Tuple contains the new: (new Position, None, position Shift/ force)

        """

        # integrate
        # while no value in spaceRange was found, terminates in first run if no spaceRange
        current_state = system.currentState
        self.oldpos = current_state.position
        
        while(True):
            self.randomShift(system.nDim)
            self.newPos = np.add(self.oldpos,self.posShift)

            #only get positions in certain range or accept if no range
            if(self._critInSpaceRange(self.newPos)):
                break

        if(self.verbose):
            print(str(self.__name__)+": current position\t ", self.oldpos)
            print(str(self.__name__)+": shift\t ", self.posShift)
            print(str(self.__name__)+": newPosition\t ", self.newPos)
            print("\n")

        return np.squeeze(self.newPos), np.nan, np.squeeze(self.posShift)
    


class metropolisMonteCarloIntegrator(stochasticIntegrator):
    """
    metropolisMonteCarloIntegrator 
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
    
    #Parameters:
    metropolisCriterion=None    #use a different Criterion
    randomnessIncreaseFactor:float = 1  #tune randomness of your results
    maxIterationTillAccept:float = np.inf  #how often shall the integrator iterate till it accepts a step forcefully
    convergence_limit:int=1000  # after reaching a certain limit abort iteration

    #METROPOLIS CRITERION
    ##random part of Metropolis Criterion:
    _defaultRandomness = lambda self, ene_new, currentState: ((1/self.randomnessIncreaseFactor)*np.random.rand() <= np.exp(-1.0 / (const.gas_constant / 1000.0 * currentState.temperature) * (ene_new - currentState.totPotEnergy)))
    ##default Metropolis Criterion
    _defaultMetropolisCriterion = lambda self, ene_new, currentState: (ene_new < currentState.totEnergy or self._defaultRandomness(ene_new, currentState))

    def __init__(self, minStepSize:float=None, maxStepSize:float=1, spaceRange:tuple=None, fixedStepSize=None, 
                metropolisCriterion=None, randomnessIncreaseFactor=1, maxIterationTillAccept:int=np.inf):
        """
        __init__ 
            This is the Constructor of the Metropolis-MonteCarlo integrator.


        Parameters
        ----------
        maxStepSize : Number, optional
            maximal size of an integrationstep in any direction, by default 1
        minStepSize : Number, optional
            minimal size of an integration step in any direction, by default None
        spaceRange : Tuple[Number, Number], optional
            maximal and minimal allowed position for after an integration step. 
            If not fullfilled, step is rejected. By default None
        fixedStepSize : Number, optional
            this option restrains each integration step to a certain size in each dimension, by default None

        metropolisCriterion : Callable, optional
            The metropolis criterion deciding if a step is accepted if None.
            But can be adapted by user providing a function as argument. By default None
        randomnessIncreaseFactor : int, optional
            arbitrary factor, controlling the ammount of randomness(the bigger the more random steps), by default 1
        maxIterationTillAccept : int, optional
            number, after which a step is accepted, regardless its likelihood (turned off if np.inf). By default None
        """

        #Integration Step Constrains
        self.fixedStepSize = None if(isinstance(fixedStepSize, type(None))) else np.array(fixedStepSize)
        self.maxStepSize = maxStepSize
        self.minStepSize = minStepSize
        self.spaceRange = spaceRange


        #Metropolis Criterions
        self.randomnessIncreaseFactor = randomnessIncreaseFactor
        self.maxIterationTillAccept = maxIterationTillAccept
        self.convergence_limit = self.convergence_limit if(isinstance(maxIterationTillAccept, type(None))) else maxIterationTillAccept+1
        
        if(metropolisCriterion == None):
            self.metropolisCriterion = self._defaultMetropolisCriterion
        else:
            self.metropolisCriterion = metropolisCriterion


    def step(self, system:systemType)-> Tuple[float, None, float]:
        """
        step 
            This function is performing an Metropolis Monte Carlo integration step.

        Parameters
        ----------
        system : systemType
            A system, that should be integrated.

        Returns
        -------
        Tuple[float, None, float]
            This Tuple contains the new: (new Position, None, position Shift/ force)

        """

        current_iteration = 0
        current_state = system.currentState
        self.oldpos = current_state.position
        nDim = system.nDim
        
        # integrate position
        while(current_iteration <= self.convergence_limit and current_iteration<=self.maxIterationTillAccept):    #while no value in spaceRange was found, terminates in first run if no spaceRange
            self.randomShift(nDim)
            #eval new Energy
            system._currentPosition = np.add(self.oldpos, self.posShift)
            system._currentForce = self.posShift
            ene = system.totPot()

            #MetropolisCriterion
            if ((self._critInSpaceRange(system._currentPosition) and self.metropolisCriterion(ene, current_state))):
                break
            else:   #not accepted
                current_iteration += 1
                continue
        if(current_iteration >= self.convergence_limit):
            raise ValueError("Metropolis-MonteCarlo integrator did not converge! Think about the maxIterationTillAccept")

        self.newPos=self.oldpos
        if(self.verbose):
            print(str(self.__name__)+": current position\t ", self.oldpos)
            print(str(self.__name__)+": shift\t ", self.posShift)
            print(str(self.__name__)+": newPosition\t ", self.newPos)
            print(str(self.__name__)+": iteration "+str(current_iteration)+"/"+str(self.convergence_limit))
            print("\n")

        return np.squeeze(system._currentPosition), np.nan, np.squeeze(self.posShift)

'''
Langevin stochastic integration
'''

class langevinIntegrator(_integratorCls):


    def __init__(self, dt:float=0.005, gamma:float=50, oldPosition:float=None):
        """
          __init__
              This is the Constructor of the Langevin integrator.


          Parameters
          ----------
          dt : Number, optional
              time step of an integration, by default 0.005
          gamma : Number, optional
              Friktion constant of the system
          oldPosition : Iterable[Number, Number] of size nDim, optional
              determins position at step -1, if not set the system will use the velocity to determine tis position
          """

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
        self.currentPosition = np.array(system._currentPosition)
        self.currentVelocity = np.array(system._currentVelocities)

        # hirachy: first check if old postition is given, if not it takes the velocity from the system class
        # is there no initial velocity a Maxwell-Boltzmann distributied velocity is generated
        if self._oldPosition is None:
            # get old position from velocity, only during initialization
            print("initializing Langevin old Positions\t ")
            print("\n")
            self._oldPosition = self.currentPosition - self.currentVelocity * self.dt
        else:
            self._oldPosition = np.array(self._oldPosition)

        # integration step
        new_position, new_velocity = self.update_positon(system)
        # update position
        self._oldPosition = self.currentPosition



        if(self.verbose):
            print(str(self.__name__)+": current forces\t ", self.newForces)
            print(str(self.__name__)+": old Position\t ", sef._oldPosition)
            print(str(self.__name__)+": current_position\t ", currentPosition)
            print(str(self.__name__)+": current_velocity\t ", currentVelocity)
            print(str(self.__name__)+": newPosition\t ", new_position)
            print(str(self.__name__)+": newVelocity\t ", new_velocity)
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
