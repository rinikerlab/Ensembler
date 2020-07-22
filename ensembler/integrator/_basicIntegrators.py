import numpy as np
import scipy.constants as const
#
from ensembler.util.ensemblerTypes import system as systemType, Tuple

class _integratorCls:
    """
            This class is the parent class for all integrator classes. 
            The constructor is a interface method. 
            Each subclass should implement it's own constructor.  

    Raises
    ------
    NotImplementedError
        These functions need to be implemented by the subclass

    """
    #general:
    verbose:bool = False
    nDim:int = 0

    
    def __init__(self):
        """
            This is a default constructor

        Raises
        ------
        NotImplementedError
            You need to implement this function in the subclass (i.e. in your integrator)
        """
        raise NotImplementedError("This "+str(__class__)+" class is not implemented")
    
    def step(self, system:systemType)->Tuple[float, float, float]:
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
        raise NotImplementedError("The function step in "+str(__class__)+" class is not implemented")

    def integrate(self, system:systemType, steps:int)->None:
        """
        integrate This function provides an alternative way for System.simulate, just executed by the integrator class.

        Parameters
        ----------
        system : systemType
            The system that should be integrated
        steps : int
            Ammount of integration steps.
        """
        
        for step in range(steps):
            (newPosition, newVelocity, newForces) = self.step(system=system)
            system.append_state(newPosition=newPosition, newVelocity=newVelocity, newForces=newForces)

    def setVerbose(self, verbose:bool=True):
        """
        setVerbose this function sets the verbosity flag of the class.

        Parameters
        ----------
        verbose : bool, optional
            set verbosity with this value. If true, it can get loud. By default True
        """
        
        self.verbose = verbose
