import numpy as np, sympy as sp

from ensembler.util.basic_class import _baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import Iterable, Union, Dict, Number

# from concurrent.futures.thread import ThreadPoolExecutor

class _potentialCls(_baseClass):
    """
    potential base class - the mother of all potential classes (or father).

    @nullState
    @Strategy Pattern
    """

    #PRIVATE ATTRIBUTES
    __nDimensions: sp.Symbol = sp.symbols("nDimensions")   #this Attribute gives the symbol of dimensionality of the potential, please access via nDimensions
    __nStates: sp.Symbol = sp.symbols("nStates") # this Attribute gives the ammount of present states(interesting for free Energy calculus), please access via nStates
    # __threads: int = 1  #Not satisfyingly implemented

    def __init__(self, nDimensions:int=1, nStates:int=2):
        if(not hasattr(self, "_potentialCls__constants")): self.__constants: Dict[sp.Symbol, Union[Number, Iterable]] = {}#contains all set constants and values for the symbols of the potential function, access it via constants
        self.__constants.update({self.nDimensions: nDimensions, self.nStates: nStates})
        self.name = str(self.__class__.__name__)

    """
        Non-Class Attributes
    """
    @property
    def nDimensions(self)->sp.Symbol:
        """
        #The symbol for the equation representing the dimensionality.
        #@Immutable!
        """
        return self.__nDimensions

    @property
    def nStates(self)->sp.Symbol:
        """
        The symbol for the equation representing the number of states. Used in Free Energy Calculations.
        @Immutable!
        """
        return self.__nStates

    @property
    def constants(self) -> dict:
        """
        This Attribute is giving all the necessary Constants to a function
        """
        return self.__constants

    @constants.setter
    def constants(self, constants: Dict[sp.Symbol, Union[Number, Iterable]]):
        self.__constants = constants


class _potentialNDCls(_potentialCls):
    '''
    Potential Base Class for N-Dimensional equations and lower ones

    @Strategy Pattern
    '''
    position : sp.Symbol
    V_functional: sp.Function = notImplementedERR
    dVdpos_functional: sp.Function = notImplementedERR

    V: sp.Function = notImplementedERR
    dVdpos = notImplementedERR

    def __init__(self, nDimensions: int = -1, nStates: int = 1):
        """
            __init__
                This class constructs the potential class basic functions, initializes the functions if necessary and also does simplfy and derivate the symbolic equations.
        Parameters
        ----------
        nDimensions: int, optional
            number of dimensions of the potential
        nStates: int, optional
            number of states in the potential.
        """

        super().__init__(nDimensions=nDimensions, nStates=nStates)

        # needed for multi dim functions to be generated dynamically
        self._initialize_functions()
        # apply potential simplification and update calc functions
        self._update_functions()

    def __str__(self) -> str:
        """
        This function converts the information of the potential class into a string.
        """
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.constants[self.nStates]) + "\n"
        msg += "\tDimensions: " + str(self.constants[self.nDimensions]) + "\n"
        msg += "\n\tFunctional:\n "
        msg += "\t\tV:\t" + str(self.V_functional) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos_functional) + "\n"
        msg += "\n\tSimplified Function\n"
        msg += "\t\tV:\t" + str(self.V) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos) + "\n"
        msg += "\n\tConstants: \n\t\t" + "\n\t\t".join(
            [str(keys) + ": " + "\n".join(["\t\t\t" + v for v in str(values).split("\n")]) for keys, values in
             self.constants.items()]) + "\n"
        msg += "\n"
        return msg

    def __setstate__(self, state):
        """
        Setting up after pickling.
        """
        self.__dict__ = state
        self._initialize_functions()
        self._update_functions()

    """
        private
    """

    def _initialize_functions(self):
        """
        This function is needed if the functions need to be adapted to the dimensionality for example
        """
        notImplementedERR()

    def _update_functions(self):
        """
        This function is needed to simplyfiy the symbolic equation on the fly and to calculate the position derivateive.
        """

        self.V = self.V_functional.subs(self.constants).expand() # expand does not work reliably with gaussians due to exp

        self.dVdpos_functional = sp.diff(self.V_functional, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")

    """
        public
    """
    _calculate_energies = lambda x: notImplementedERR() #is generated by update_functions()
    _calculate_dVdpos = lambda x: notImplementedERR()#is generated by update_functions()

    def ene(self, positions: Union[Number, Iterable[Number], Iterable[Iterable[Number]]]) -> Union[Number, Iterable[Number]]:
        """
            ene
                calculates the potential energy of the given position/s using the potential function.

        Parameters
        ----------
        positions: Union[Number, Iterable]

        Returns
        -------
        ene: Union[Number, Iterable]
            the calculated potential energies.

        """
        return np.squeeze(self._calculate_energies(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions])))

    def force(self, positions:Union[Number, Iterable[Number], Iterable[Iterable[Number]]]) -> Union[Number, Iterable[Number], Iterable[Iterable[Number]]]:
        """
            force
                calculates the potential forces/gradients of the given position/s using the derivative of potential function with the position.

        Parameters
        ----------
        positions: Union[Number, Iterable]

        Returns
        -------
        force: Union[Number, Iterable]
            the calculated potential forces.

        """
        return np.squeeze(self._calculate_dVdpos(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions]))).T

    # just alternative name, same as force
    def dvdpos(self, positions:Union[Number, Iterable[Number], Iterable[Iterable[Number]]]) -> Union[Number, Iterable[Number], Iterable[Iterable[Number]]]:
        return self.force(positions)


class _potential1DCls(_potentialNDCls):
    '''
    Potential Base Class for 1-Dimensional equations

    @Strategy Pattern
    '''
    def __init__(self, nStates: int = 1):
        """
            __init__
                constructs a N-Dimensional class with nDimensions =1

        Parameters
        ----------
        nStates: int, optional
            number of states in the potential.
        """
        super().__init__(nDimensions=1, nStates=nStates)

    def _initialize_functions(self):
        """
        Normally not needed in the one dimensional case
        """
        pass

    def ene(self, positions: Union[Number, Iterable[Number]]) -> Union[Number, Iterable[Number]]:
        """
            ene
                calculates the potential energy of the given position/s using the potential function.

        Parameters
        ----------
        positions: Union[Number, Iterable]

        Returns
        -------
        ene: Union[Number, Iterable]
            the calculated potential energies.

        """
        return np.squeeze(self._calculate_energies(np.array(positions)))

    def force(self, positions: Union[Iterable[Number] or Number]) -> Union[Iterable[Number] or Number]:
        """
            force
                calculates the potential forces/gradients of the given position/s using the derivative of potential function with the position.

        Parameters
        ----------
        positions: Union[Number, Iterable]

        Returns
        -------
        force: Union[Number, Iterable]
            the calculated potential forces.

        """
        return np.squeeze(self._calculate_dVdpos(np.squeeze(np.array(positions))))


class _potential2DCls(_potentialNDCls):
    '''
    Potential Base Class for 2-Dimensional equations

    @Strategy Pattern
    '''
    def __init__(self, nStates: int = 1):
        """
            __init__
                constructs a N-Dimensional class with nDimensions =1

        Parameters
        ----------
        nStates: int, optional
            number of states in the potential. (default: 1)
        """
        super().__init__(nDimensions=2, nStates=nStates)


class _potential1DClsPerturbed(_potential1DCls):
    '''
    Potential Base Class for 1-Dimensional potential functions, that are coupled as linear combination.

    @Strategy Pattern
    '''

    coupling: sp.Function = notImplementedERR

    lam = sp.symbols(u"λ")
    statePotentials: Dict[sp.Function, sp.Function]

    dVdlam_functional: sp.Function
    dVdlam = notImplementedERR


    def __str__(self) -> str:
        """
        This function converts the information of the perturbed potential class into a string.
        """
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.constants[self.nStates]) + "\n"
        msg += "\tDimensions: " + str(self.nDimensions) + "\n"
        msg += "\n\tFunctional:\n "
        msg += "\t\tCoupling:\t" + str(self.coupling) + "\n"
        msg += "\t\tV:\t" + str(self.V_functional) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos_functional) + "\n"
        msg += "\t\tdVdlam:\t" + str(self.dVdlam_functional) + "\n"
        msg += "\n\tSimplified Function\n"
        msg += "\t\tV:\t" + str(self.V) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos) + "\n"
        msg += "\t\tdVdlam:\t" + str(self.dVdlam) + "\n"
        msg += "\n\tConstants: \n\t\t" + "\n\t\t".join(
            [str(keys) + ": " + "\n".join(["\t\t\t" + v for v in str(values).split("\n")]) for keys, values in
             self.constants.items()]) + "\n"
        msg += "\n"
        return msg

    """
        private
    """

    def _update_functions(self):
        """
        This function sets the coupling as functional and builds the dVdlam derivateive.
        Returns
        -------

        """
        self.V_functional = self.coupling

        super()._update_functions()

        self.dVdlam_functional = sp.diff(self.V_functional, self.lam)
        self.dVdlam = self.dVdlam_functional.subs(self.constants)
        self._calculate_dVdlam = sp.lambdify(self.position, self.dVdlam, "numpy")

    """
        public
    """

    def set_lambda(self, lam: Number):
        """
        set the lambda paramter, coupling the states of the system.

        Parameters
        ----------
        lam: float
            normally a value between 0 and 1, where 0 is representing on stateA and 1 the second state B
        """
        self.constants.update({self.lam: lam})
        self._update_functions()

    def lambda_force(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        """
            dvdlam
                calculates the potential forces/gradients of the given position/s using the derivative of potential function with the lambda paramter.

        Parameters
        ----------
        positions: Union[Number, Iterable]

        Returns
        -------
        lambda_force: Union[Number, Iterable]
            the calculated potential lambda_forces.

        """
        return np.squeeze(self._calculate_dVdlam(np.squeeze(positions)))

    #just a different name
    def dvdlam(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        return self.lambda_force(positions=positions)
