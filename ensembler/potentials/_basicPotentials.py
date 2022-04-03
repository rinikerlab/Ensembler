import copy
import pint
import numpy as np, sympy as sp
from ensembler.util.ensemblerTypes import Iterable, Union, Dict, Number

from ensembler.util.basic_class import _baseClass, notImplementedERR
from ensembler.util.ensemblerTypes import Number, Union, List, Dict, Iterable
from ensembler.util.units import quantity

# from concurrent.futures.thread import ThreadPoolExecutor


class _potentialCls(_baseClass):
    """
        potential base class - the mother of all potential classes (or father), it gives the coarse structure to what a potential class needs.

    @nullState
    @Strategy Pattern
    """

    # PRIVATE ATTRIBUTES
    __nDimensions: sp.Symbol = sp.symbols(
        "nDimensions"
    )  # this Attribute gives the symbol of dimensionality of the potential, please access via nDimensions
    __nStates: sp.Symbol = sp.symbols(
        "nStates"
    )  # this Attribute gives the ammount of present states(interesting for free Energy calculus), please access via nStates
    __constants: Dict[sp.Symbol, Union[Number, pint.Quantity]]  # in this dictionary all constants are stored.
    # __threads: int = 1  #Not satisfyingly implemented

    # toward Private
    _unitless: bool

    def __init__(self, nDimensions: int = 1, nStates: int = 2, unitless: bool = False):
        if not hasattr(self, "_potentialCls__constants"):
            self.__constants: Dict[
                sp.Symbol, Union[Number, Iterable]
            ] = {}  # contains all set constants and values for the symbols of the potential function, access it via constants
        self.__constants.update({self.nDimensions: nDimensions, self.nStates: nStates})
        self.name = str(self.__class__.__name__)
        self._unitless = unitless

    """
        Non-Class Attributes
    """

    @property
    def nDimensions(self) -> sp.Symbol:
        """
        #The symbol for the equation representing the dimensionality.
        #@Immutable!
        """
        return self.__nDimensions

    @property
    def nStates(self) -> sp.Symbol:
        """
        The symbol for the equation representing the number of states. Used in Free Energy Calculations.
        @Immutable!
        """
        return self.__nStates

    @property
    def constants(self) -> Dict[sp.Symbol, Union[Number, pint.Quantity]]:
        """
        This Attribute is giving all the necessary Constants to a function
        """
        return self.__constants

    @constants.setter
    def constants(self, constants: Dict[sp.Symbol, Union[Number, Iterable]]):
        self.__constants = constants

    @property
    def unitless(self) -> bool:
        return self._unitless


class _potentialNDCls(_potentialCls):
    """
        Potential Base Class for N-Dimensional equations and lower ones - This classes is intended to use symbolic functionals with SymPy,
        that are translated to efficient numpy functions.

    @Strategy Pattern
    """

    position: sp.Symbol  # This sympol is required and resembles the input parameter (besides the constants).
    V_functional: sp.Function = notImplementedERR  # This attribute needs to be provided.
    dVdpos_functional: sp.Function = notImplementedERR  # This function will be generated in the constructure by initialize-functions

    V: sp.Function = (
        notImplementedERR  # This function will be generated in the constructure by initialize-functions - simplified symbolic function
    )
    dVdpos = (
        notImplementedERR  # This function will be generated in the constructure by initialize-functions  - simplified symbolic function
    )
    dimensionless_constants: List[sp.Symbol]  # These are the expected dimensionless constants, that will be used.

    # Todo: nice error message for the expected output units.
    def __init__(self, nDimensions: int = -1, nStates: int = 1, unitless: bool = False):
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
        self.dimensionless_constants = [nDimensions, nStates]

        super().__init__(nDimensions=nDimensions, nStates=nStates, unitless=unitless)

        # Do unit managment if required:
        self._constant_unit_management()
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
        msg += (
            "\n\tConstants: \n\t\t"
            + "\n\t\t".join(
                [str(keys) + ": " + "\n".join(["\t\t\t" + v for v in str(values).split("\n")]) for keys, values in self.constants.items()]
            )
            + "\n"
        )
        msg += "\n"
        return msg

    def __setstate__(self, state):
        """
        Setting up after pickling.
        """
        self.__dict__ = state
        self._constant_unit_management()
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

    def _constant_unit_management(self):
        """
        This function, splits magnitudes and units depending on desired case

        """
        get_magnitude = lambda x: x.magnitude if (isinstance(x, quantity)) else x

        def get_unit(x):  # we need to symbolize pint units here again, as they are not correctly handled otherwise.
            if isinstance(x, quantity):
                return 1 * x.units
            elif x in self.dimensionless_constants:
                return x
            else:
                return 1

        if self.unitless:
            self._constants_units = {k: 1 for k, _ in self.constants.items()}
            self._constants_magnitude = {k: get_magnitude(v) for k, v in self.constants.items()}

        else:
            self._constants_units = {k: get_unit(v) for k, v in self.constants.items()}
            self._constants_magnitude = {k: get_magnitude(v) for k, v in self.constants.items()}

    def _update_functions(self):
        """
        This function is needed to simplyfiy the symbolic equation on the fly and to calculate the position derivateive.
        """

        self.V = self.V_functional.subs(self._constants_magnitude).expand()  # expand does not work reliably with gaussians due to exp
        if not self._unitless:
            self.V_units = self.V_functional.subs(self._constants_units)

        self.dVdpos_functional = sp.diff(self.V_functional, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)
        if not self._unitless:
            self.dVdpos_units = self.dVdpos_functional.subs(self._constants_units)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")

    def _solve_units(self, functional, position_unit):
        """
        Note: we need to symbolize pint units here again, as they are not correctly handled otherwise.
        """

        argsKV = list(self._constants_units.items())
        argsK = [k for k, v in argsKV]
        argsV = [1 * v for k, v in argsKV]

        if self.constants[self.nDimensions] > 1:
            for p in self.position:
                argsK.append(p)
                argsV.append(1 * position_unit)
        else:
            argsK.append(self.position)
            argsV.append(1 * position_unit)

        print(argsK, argsV)
        self.argsK = tuple(argsK)
        self.argsV = tuple(argsV)
        self.ff = sp.lambdify(self.argsK, self.V_functional, "numpy")
        self._solved_units = self.ff(*argsV)
        print(self._solved_units, type(self._solved_units))

        return self._solved_units.units
        # except Exception as err:
        #    raise Exception("The unit Conversion did not work! Please check if all variables have the correct units.\n"+
        #                    "SymPy solving try: "+str(self._solved_units))

    """
        public
    """
    _calculate_energies = lambda x: notImplementedERR()  # is generated by update_functions()
    _calculate_dVdpos = lambda x: notImplementedERR()  # is generated by update_functions()

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
        if self._unitless:
            return np.squeeze(self._calculate_energies(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions])))
        else:
            if isinstance(positions, quantity):
                return quantity(
                    value=np.squeeze(self._calculate_energies(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions]))),
                    units=self._solve_units(functional=self.V_units, position_unit=positions.units),
                )
            else:
                raise ValueError("Your positions don't have a unit! (and you are not in unitless mode. Please add a unit.")

    def force(
        self, positions: Union[Number, Iterable[Number], Iterable[Iterable[Number]]]
    ) -> Union[Number, Iterable[Number], Iterable[Iterable[Number]]]:
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
        if hasattr(positions, "dimensionless") and not positions.dimensionless and not self._unitless:
            return quantity(
                value=np.squeeze(self._calculate_dVdpos(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions]))).T,
                units=self._solve_units(functional=self.dVdpos_units, position_unit=positions.units),
            )
        else:
            return np.squeeze(self._calculate_dVdpos(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions]))).T

    # just alternative name, same as force
    def dvdpos(
        self, positions: Union[Number, Iterable[Number], Iterable[Iterable[Number]]]
    ) -> Union[Number, Iterable[Number], Iterable[Iterable[Number]]]:
        return self.force(positions)


class _potential1DCls(_potentialNDCls):
    """
        Potential Base Class for 1-Dimensional equations

    @Strategy Pattern
    """

    def __init__(self, nStates: int = 1, unitless: bool = False):
        """
            __init__
                constructs a N-Dimensional class with nDimensions =1

        Parameters
        ----------
        nStates: int, optional
            number of states in the potential.
        """
        super().__init__(nDimensions=1, nStates=nStates, unitless=unitless)

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
        if self._unitless:
            return np.squeeze(self._calculate_energies(np.array(positions)))
        else:
            if isinstance(positions, quantity):
                return quantity(
                    value=np.squeeze(self._calculate_energies(np.array(positions.magnitude))),
                    units=self._solve_units(functional=self.V_units, position_unit=positions.units),
                )
            else:
                raise ValueError(
                    "Your positions don't have a unit! (and you are not in unitless mode. Please add a unit. \n\t Hint: we are looking for the missing unit -  "
                    + str(self.V_units)
                )

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
        if hasattr(positions, "dimensionless") and not positions.dimensionless and not self._unitless:
            return quantity(
                value=np.squeeze(self._calculate_dVdpos(np.array(positions.magnitude))),
                units=self._solve_units(functional=self.dVdpos_units, position_unit=positions.units),
            )
        else:
            return np.squeeze(self._calculate_dVdpos(np.squeeze(np.array(positions))))


class _potential2DCls(_potentialNDCls):
    """
    Potential Base Class for 2-Dimensional equations

    @Strategy Pattern
    """

    def __init__(self, nStates: int = 1, unitless: bool = False):
        """
            __init__
                constructs a N-Dimensional class with nDimensions =1

        Parameters
        ----------
        nStates: int, optional
            number of states in the potential. (default: 1)
        """
        print("Warning no units for 2D right now!")
        super().__init__(nDimensions=2, nStates=nStates, unitless=unitless)


class _potential1DClsPerturbed(_potential1DCls):
    """
    Potential Base Class for 1-Dimensional potential functions, that are coupled as linear combination.

    @Strategy Pattern
    """

    coupling: sp.Function = notImplementedERR

    lam = sp.symbols("λ")
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
        msg += (
            "\n\tConstants: \n\t\t"
            + "\n\t\t".join(
                [str(keys) + ": " + "\n".join(["\t\t\t" + v for v in str(values).split("\n")]) for keys, values in self.constants.items()]
            )
            + "\n"
        )
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

        super()._constant_unit_management()
        super()._update_functions()

        self.dVdlam_functional = sp.diff(self.V_functional, self.lam)
        self.dVdlam = self.dVdlam_functional.subs(self._constants_magnitude)
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

    # just a different name
    def dvdlam(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        return self.lambda_force(positions=positions)
