"""
Module: System
    This module shall be used to implement subclasses of system1D. It wraps all information needed and generated by a simulation.
"""

import os, numpy as np
from tqdm.notebook import tqdm
from typing import Iterable, NoReturn, List, Sized
from numbers import Number
import pandas as pd
import scipy.constants as const
import warnings
pd.options.mode.use_inf_as_na = True

#Typing
import ensembler.util.ensemblerTypes as ensemblerTypes
_integratorCls = ensemblerTypes.integrator
_conditionCls = ensemblerTypes.condition
_potentialCls = ensemblerTypes.potential

from ensembler.util import dataStructure as data

from ensembler.integrator.newtonian import newtonianIntegrator
from ensembler.integrator import stochastic

class system:
    """
     [summary]
    
    :raises IOError: [description]
    :raises Exception: [description]
    :return: [description]
    :rtype: [type]
    """
    #static attributes
    state = data.basicState


    """
    Attributes:
    """
    ##POTENTIAL CLASS
    @property
    def potential(self)-> _potentialCls:
        return self.m_potential
    
    @potential.setter
    def potential(self, potential:_potentialCls):
        # if(issubclass(potential.__class__, _potentialCls)):
        self.m_potential = potential
        # else:
        #     raise ValueError("Potential needs to be a subclass of potential")

    @property
    def integrator(self)->_integratorCls:
        return self.m_integrator
    
    @integrator.setter
    def integrator(self, integrator:_integratorCls):
        self.m_integrator = integrator
   
    @property
    def conditions(self)->List[_conditionCls]:
        return self.m_conditions
    
    @conditions.setter
    def conditions(self, conditions:List[_conditionCls]):
        if(isinstance(conditions, List) and all([issubclass(condition.__class__, _conditionCls) for condition in conditions])):
            self.m_conditions = conditions
        else:
            raise ValueError("Conditions needs to be a List of objs, that are a subclass of _conditionCls")
    
    def __init__(self, potential:_potentialCls, integrator:_integratorCls, conditions:Iterable[_conditionCls]=[],
                 temperature:Number=298.0, position:(Iterable[Number] or Number)=None, mass:Number=1, verbose:bool=True)->NoReturn:
        ################################
        # Declare Attributes
        #################################
    
        ##Physical parameters
        self.temperature: float = temperature
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


        #BUILD System
        ## Fundamental Parts:
        self.potential = potential
        self.integrator = integrator
        self.conditions = conditions

        ## set dim
        if(potential.constants[potential.nDim] < 1 and isinstance(position, Iterable) and all([isinstance(pos, Number) for pos in position])):  #one  state system.
            self.nDim = len(position)
            self.potential.constants.update({potential.nDim: self.nDim})
        elif(potential.constants[potential.nDim] > 0):
            self.nDim = potential.constants[potential.nDim]
        else:
            raise IOError("Could not estimate the disered Dimensionality as potential dim was <1 and no initial position was given.")
        self.mass = mass

        ###is the potential a state dependent one? - needed for initial pos.
        if(hasattr(potential, "nStates")):
            self.nStates = potential.constants[potential.nStates]
            if(hasattr(potential, "states_coupled")):   #does each state get the same position?
                self.states_coupled = potential.states_coupled
            else:
                self.states_coupled = True #Todo: is this a good Idea?
        else:
            self.nstates = 1

        #PREPARE THE SYSTEM
        #Only init velocities, if the integrator uses them
        if(issubclass(integrator.__class__, (newtonianIntegrator, stochastic.langevinIntegrator))) :
            init_velocity=True
        else:
            init_velocity=False
        self.initialise(withdraw_Traj=True, init_position=True, init_velocity=init_velocity, set_initial_position=position)

        ##check if system should be coupled to conditions:
        for condition in self.conditions:
            if(not hasattr(condition, "system")):
                condition.coupleSystem(self)
            else:
                #warnings.warn("Decoupling system and coupling it again!")
                condition.coupleSystem(self)
            if(not hasattr(condition, "dt") and hasattr(self.integrator, "dt")):
                condition.dt = self.integrator.dt
            else:
                condition.dt=1

        self.verbose = verbose

    """
        Initialisation
    """
    def initialise(self, withdraw_Traj:bool=True, init_position:bool=True, init_velocity:bool=True, set_initial_position=None):
        if(withdraw_Traj):
            self.trajectory = pd.DataFrame(columns=list(self.state.__dict__["_fields"]))

        if(init_position):
            self._init_position(initial_position=set_initial_position)

        #Try to init the force
        try:
            self._currentForce = self.potential.dvdpos(self.initial_position)  #initialise forces!    #todo!
        except:
            warnings.warn("Could not initialize the force of the potential? Check if you need it!")

        if(init_velocity):
            self._init_velocities()

        # set initial Temperature
        self._currentTemperature = self.temperature

        #update current state
        self.updateSystemProperties()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def _init_position(self, initial_position=None):
        if (isinstance(initial_position, type(None))):
            self.initial_position = self.randomPos()
        elif(isinstance(initial_position, (Number,Iterable))):
            self.initial_position = initial_position
        else:
            raise Exception("did not understand the initial position!")

        self._currentPosition = self.initial_position
        self.updateCurrentState()
        return self.initial_position

    def _init_velocities(self)-> NoReturn:
        if(self.nStates>1):
            self._currentVelocities = [[self._gen_rand_vel() for dim in range(self.nDim)] for s in range(self.nStates)] if(self.nDim>1) else [self._gen_rand_vel() for state in range(self.nStates)]
        else:
            self._currentVelocities = [self._gen_rand_vel() for dim in range(self.nDim)] if (self.nDim > 1) else self._gen_rand_vel()

        self.veltemp = self.mass / const.gas_constant / 1000.0 * np.linalg.norm(self._currentVelocities) ** 2  # t

        self.updateCurrentState()
        return self._currentVelocities

    def _gen_rand_vel(self)->float:
        return np.sqrt(const.gas_constant / 1000.0 * self.temperature / self.mass) * np.random.normal()

    def randomPos(self)-> (np.array or np.float):
        """uncoupled states
        if(self.nStates > 1):
            return [np.subtract(np.multiply(np.random.rand(self.nDim),20),10) for state in range(self.nStates)]
        else:
        """
        random_pos = np.squeeze(np.array(np.subtract(np.multiply(np.random.rand(self.nDim), 20), 10)))
        if(len(random_pos.shape) == 0):
            return np.float(random_pos)
        else:
            return random_pos


    """
        Update
    """
    def totKin(self)-> (Iterable[Number] or Number or None):
        # Todo: more efficient if?
        if(self.nDim == 1 and isinstance(self._currentVelocities, Number) and not np.isnan(self._currentVelocities)):
            return 0.5 * self.mass * np.square(np.linalg.norm(self._currentVelocities))
        elif(self.nDim > 1 and isinstance(self._currentVelocities, Iterable) and all([isinstance(x, Number) and not np.isnan(x) for x in self._currentVelocities])):
            return np.sum(0.5 * self.mass * np.square(np.linalg.norm(self._currentVelocities)))
        else:
            return np.nan

    def totPot(self)-> (Iterable[Number] or Number or None):
        return self.potential.ene(self._currentPosition)

    def updateSystemProperties(self)-> NoReturn:
        self._updateEne()
        self._updateTemp()

    def updateCurrentState(self)-> NoReturn:
        self.currentState = self.state(self._currentPosition, self._currentTemperature,
                                        self._currentTotE, self._currentTotPot, self._currentTotKin,
                                        self._currentForce, self._currentVelocities)

    def _updateTemp(self)-> NoReturn:
        """ this looks like a thermostat like thing! not implemented!@ TODO calc velocity from speed"""
        self._currentTemperature = self._currentTemperature

    def _updateEne(self)-> NoReturn:
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()
        self._currentTotE = self._currentTotPot if(np.isnan(self._currentTotKin))else np.add(self._currentTotKin, self._currentTotPot)

    def _update_current_vars_from_current_state(self):
        self._currentPosition = self.currentState.position
        self._currentTemperature = self.currentState.temperature
        self._currentTotE = self.currentState.totEnergy
        self._currentTotPot = self.currentState.totPotEnergy
        self._currentTotKin = self.state.totKinEnergy
        self._currentForce = self.currentState.dhdpos
        self._currentVelocities = self.currentState.velocity

    """
        Functionality
    """
    def simulate(self, steps:int,  withdrawTraj:bool=False, save_every_state:int=1, initSystem:bool=False, verbosity:bool=True)-> state:

        if(withdrawTraj):
            self.trajectory: pd.DataFrame = pd.DataFrame(columns=list(self.state.__dict__["_fields"]))
            self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

        if(initSystem):
            self._init_position()
            self._init_velocities()

        self.updateCurrentState()
        self.updateSystemProperties()

        #Simulation loop
        for step in tqdm(range(steps), desc="Simulation: ", mininterval=1.0, leave=verbosity):

            #Do one simulation Step. Todo: change to do multi steps
            self.propagate()

            #Calc new Energy&and other system properties
            self.updateSystemProperties()

            #Apply Restraints, Constraints ...
            self.applyConditions()

            #Set new State
            self.updateCurrentState()

            if(step%save_every_state == 0 and step != steps-1):
                self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)
        self.potential.set_simulation_mode(False)

        return self.currentState

    def propagate(self)->NoReturn:
        self._currentPosition, self._currentVelocities, self._currentForce = self.integrator.step(self)

    def applyConditions(self)-> NoReturn:
        for aditional in self.conditions:
            #setattr(aditional, "system", self)  #todo: nicer solution?
            aditional.apply()

    def append_state(self, newPosition, newVelocity, newForces)->NoReturn:
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces

        self._updateTemp()
        self._updateEne()
        self.updateCurrentState()

        self.trajectory = self.trajectory.append(self.currentState._asdict(), ignore_index=True)

    def revertStep(self)-> NoReturn:
        self.currentState = self.trajectory.iloc[-2]
        self._update_current_vars_from_current_state()
        return

    def _update_state_from_traj(self)-> NoReturn:
        print(self.trajectory.iloc[-1].to_dict())
        self.currentState = self.state(**self.trajectory.iloc[-1].to_dict())
        self._update_current_vars_from_current_state()
        return
    """
        Getter
    """
    #Getters
    def getTotPot(self)-> (Iterable[Number] or Number or None):
        return self._currentTotPot

    def getTotEnergy(self)-> (Iterable[Number] or Number or None):
        return self._currentTotE

    def getCurrentState(self)->state:
        return self.currentState

    def getTrajectory(self)->pd.DataFrame:
        return self.trajectory

    #writing out
    def writeTrajectory(self, out_path:str)->str:
        if(not os.path.exists(os.path.dirname(out_path))):
            raise Exception("Could not find output folder: "+os.path.dirname(out_path))
        self.trajectory.to_csv(out_path, header=True)
        return out_path

    #Setters
    def set_position(self, position):
        self._currentPosition = position
        if(len(self.trajectory) == 0):
            self.initial_position = self._currentPosition
        self._updateEne()
        self.updateCurrentState()

    def set_velocities(self, velocities):
        self._currentVelocities = velocities
        self._updateEne()
        self.updateCurrentState()

    def set_current_state(self, currentPosition:(Number or Iterable), currentVelocities:(Number or Iterable)=0, currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature
        self.currentState = self.state(self._currentPosition, self._currentTemperature, np.nan, np.nan, np.nan, np.nan, np.nan)

        self._updateEne()
        self.updateCurrentState()

    def set_Temperature(self, temperature):
        """ this looks like a thermostat like thing! not implemented!@"""
        self.temperature = temperature
        self._currentTemperature = temperature
        self._updateEne()
