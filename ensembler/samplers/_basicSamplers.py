"""
Module: Sampler
    The sampler module is provides methods exploring the potential functions.
"""

from ensembler.util.basic_class import _baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import systemCls as systemType, Tuple


class _samplerCls(_baseClass):
    """
            This class is the parent class for all samplers classes.
            The constructor is a interface method.
            Each subclass should implement it's own constructor.

    Raises
    ------
    NotImplementedError
        These functions need to be implemented by the subclass

    """

    # general:
    verbose: bool = False
    nDimensions: int = 0

    def __init__(self):
        """
            This is a default constructor

        Raises
        ------
        NotImplementedError
            You need to implement this function in the subclass (i.e. in your samplers)
        """
        super().__init__()
        pass

    def step(self, system: systemType) -> Tuple[float, float, float]:
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
            You need to implement this function in the subclass (i.e. in your samplers)
        """
        notImplementedERR()

    def integrate(self, system: systemType, steps: int) -> None:
        """
        integrate This function provides an alternative way for System.simulate, just executed by the samplers class.

        Parameters
        ----------
        system : systemType
            The system that should be integrated
        steps : int
            Ammount of integration steps.
        """

        for step in range(steps):
            (newPosition, newVelocity, newForces) = self.step(system=system)
            system.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces)

    def set_verbose(self, verbose: bool = True):
        """
        setVerbose this function sets the verbosity flag of the class.

        Parameters
        ----------
        verbose : bool, optional
            set verbosity with this value. If true, it can get loud. By default True
        """

        self.verbose = verbose
