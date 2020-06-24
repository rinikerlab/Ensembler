import numpy as np, sympy as sp
from numbers import Number
from typing import Iterable, Sized, Union, Dict
from concurrent.futures.thread import ThreadPoolExecutor


class _potentialCls:
    nDim: sp.Symbol = sp.symbols("N")
    nStates: int = 1

    constants:dict = {}

    pass



"""
    OLD Scaffold classes for the potentials
"""
class _potentialNDCls(_potentialCls):
    '''
    potential base class
    @nullState
    @Strategy Pattern
    '''
    threads: int = 1
    _no_Type_check: bool = False
    _singlePos_mode: bool = False

    def __init__(self, nDim: int = -1, n_threads: int = 1):
        nDim = nDim
        self._calculate_energies = self._calculate_energies_multiPos
        self._calculate_dvdpos = self._calculate_dvdpos_multiPos
        self._check_positions_type = self._check_positions_type_multiPos

        if (n_threads > 1):
            self.thread_pool = ThreadPoolExecutor(max_workers=self.threads)
        self.threads = n_threads

        self.constants.update({self.nDim: nDim})


    def __name__(self) -> str:
        return str(self.name)

    def __str__(self):
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.nStates) + "\n"
        msg += "\tDimensions: " + str(self.constants[self.nDim]) + "\n"
        msg += "\n"
        return msg

    """
        public
    """

    def ene(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates energy of particle
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: energy
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_energies(positions)

    def dvdpos(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates derivative with respect to position
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dvdpos(positions)

    """
        private
    """
    """
            dummies
    """

    @classmethod
    def _check_positions_type(cls, positions: Union[Iterable[Number], Iterable[Iterable[Number]], Number]) -> Union[
        np.array, Number]:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:   Union[Iterable[Number], Iterable[Iterable[Number]], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        raise Exception(
            __name__ + "_Dummy Was not initialized! please call super constructor " + __class__.__name__ + "!")

    def _calculate_energies(cls, positions: Union[Iterable[Number], Iterable[Iterable[Number]], Number]) -> Union[
        np.array, Number]:
        """
            .. autofunction:: _calculate_energies

        :param positions:
        :type positions:  Union[Iterable[Number], Iterable[Iterable[Number]], Number]
        :return:
        :return type: np.array
        """
        raise Exception(
            __name__ + "_Dummy Was not initialized! please call super constructor " + __class__.__name__ + "!")

    def _calculate_dvdpos(cls, positions: Union[Iterable[Number], Iterable[Iterable[Number]], Number]) -> Union[
        np.array, Number]:
        """
            .. autofunction:: _calculate_dhdpos

        :param positions:
        :type positions:  Union[Iterable[Number], Iterable[Iterable[Number]], Number]
        :return:
        :return type: np.array
        """
        raise Exception(
            __name__ + "_Dummy Was not initialized! please call super constructor " + __class__.__name__ + "!")

    """
            type Juggeling and interface methods setting 
    """

    @classmethod
    def _check_positions_type_singlePos(cls, position: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param position: here the positions that shall be evaluated by the potential should be given.
        :type position:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        # array
        if (isinstance(position, Number)):
            return position
        elif (isinstance(position, Iterable)):
            if (cls.nDim == 1 and isinstance(position, Number)):
                return position[0]
            elif ((len(position) == cls.nDim or cls.nDim == -1) and all(
                    [isinstance(position, Number) for position in position])):  # position[dimPos]
                return np.array(position, ndmin=1)
            else:
                raise Exception(
                    "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))
        else:
            raise Exception(
                "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))

    @classmethod
    def _check_positions_type_multiPos(cls, positions: Union[Iterable[Iterable[Number]], Iterable[Number], Number],
                                       threads: int = 1) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        # array
        if (isinstance(positions, Iterable)):
            if (all([isinstance(position, Iterable) and (len(position) == cls.nDim or cls.nDim == 0) for position in
                     positions])):  # positions[[position[dimPos]]
                if (all([all([isinstance(dimPos, Number) for dimPos in position]) for position in positions])):
                    return np.array(positions, ndmin=2)
                else:
                    raise Exception()
            elif (all([isinstance(position, Number) for position in positions])):  # positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            elif (all([isinstance(position, Iterable) and all([isinstance(x, Number) for x in position]) for position in
                       positions])):  # positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            elif ((cls.nDim == 1 or cls.nDim == 0) and all(
                    [isinstance(position, Number) for position in positions])):  # positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            else:
                raise Exception(
                    "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))
        elif ((cls.nDim == 1 or cls.nDim == -1) and isinstance(positions, Number)):
            return np.array(positions, ndmin=2)
        else:
            raise Exception(
                "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))

    def _calculate_energies_singlePos(self, position: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dvdpos_singlePos(self, positions: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_energies_multiPos(self, positions: Union[
        Iterable[Iterable[Number]], Iterable[Number], Number]) -> np.array:
        """
        ..autofunction :: _calculate_energies_multiPos

        :return:  -
        """
        ene = np.array(list(map(self._calculate_energies_singlePos, positions)))
        return ene.item() if (len(ene.shape) == 1 and ene.shape[0] == 1) else ene

    def _calculate_dvdpos_multiPos(self,
                                   positions: Union[Iterable[Iterable[Number]], Iterable[Number], Number]) -> np.array:
        force = np.array(list(map(self._calculate_dvdpos_singlePos, positions)))
        return force.item() if (len(force.shape) == 1 and force.shape[0] == 1) else force

    def set_simulation_mode(self, simulation: bool = True):
        if (simulation):
            self._set_singlePos_mode()
            self._set_no_type_check()
        else:
            self._set_multiPos_mode()
            self._set_type_check()

    """
            Input - Options
                For Performance or easier use!
    """

    def _set_singlePos_mode(self):
        """
        ..autofunction :: _set_singlePos_mode

        :return:  -
        """
        self._singlePos_mode = True
        self._check_positions_type = self._check_positions_type_singlePos
        self._calculate_energies = self._calculate_energies_singlePos
        self._calculate_dvdpos = self._calculate_dvdpos_singlePos
        # print(__name__+"in _set_singlePos_mode ",self.nDim)

    def _set_multiPos_mode(self):
        """
        ..autofunction :: _set_multiPos_mode

        :return:  -
        """
        self._singlePos_mode = False
        self._check_positions_type = self._check_positions_type_multiPos
        self._calculate_energies = self._calculate_energies_multiPos
        self._calculate_dvdpos = self._calculate_dvdpos_multiPos

    def _set_no_type_check(self):
        """
        ..autofunction :: _set_no_type_check
            This function is trying to speed up execution for cases, in that the position Type is known to be correct (system integration) ...

        :return:  -
        """
        _no_Type_check = True
        self._check_positions_type = lambda x: x

    def _set_type_check(self):
        """
        ..autofunction :: _set_type_check
            This function is setting the default potential Value, to allow secure execution in small code snippets
        :return:  -
        """
        _no_Type_check = False
        if (self._singlePos_mode):
            self._check_positions_type = self._check_positions_type_singlePos
        else:
            self._check_positions_type = self._check_positions_type_multiPos


class _potential1DCls(_potentialNDCls):
    '''
        .. autoclass:: _potentialCls
        This class is the
    '''

    def __init__(self):
        super().__init__(nDim=self.nDim)

    @classmethod
    def _check_positions_type_singlePos(cls, position: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param position: here the positions that shall be evaluated by the potential should be given.
        :type position:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        if (isinstance(position, Number)):
            return position
        elif (isinstance(position, Sized) and len(position) == cls.nDim and isinstance(position[0], Number)):
            return position[0]
        else:
            raise Exception(
                "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))

    @classmethod
    def _check_positions_type_multiPos(cls, positions: Union[
        Iterable[Iterable[Number]], Iterable[Number], Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """

        if (isinstance(positions, Number)):  # single number
            return np.array(positions, ndmin=1)
        elif (isinstance(positions, Iterable) and isinstance(positions, Sized)):
            if (all([isinstance(x, Number) for x in positions])):  # list with numbers
                return np.array(positions, ndmin=1)
            elif ((type(positions) != type(None) and len(positions) == 1) and all(
                    [isinstance(pos, Number) for pos in positions[0]])):
                return np.array(positions[0], ndmin=1)
            else:
                raise Exception(
                    "list dimensionality does not fit to potential dimensionality! len(list)=2 potential Dimensions 1")
        else:
            if (type(positions) == type(None)):
                raise Exception("potential got None as position")
            else:
                raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=" + str(
                    len(positions)) + " potential Dimensions " + str(cls.nDim))

    def _calculate_energies_singlePos(self, position: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dvdpos_singlePos(self, positions: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")


class _potential2DCls(_potentialNDCls):
    '''
    potential base class
    '''
    nDim: int = 2

    def __init__(self):
        super().__init__(nDim=2)

    @classmethod
    def _check_positions_type_singlePos(cls, position: Union[Iterable[Number], Number]) -> np.array:
        positions = super()._check_positions_type_singlePos(position=position)
        if (len(positions) == cls.nDim):
            return position
        else:
            raise Exception("Dimensionality is not correct for positions! " + str(positions))

    @classmethod
    def _check_positions_type_multiPos(cls, positions: Union[
        Iterable[Iterable[Number]], Iterable[Number], Number]) -> np.array:
        positions = super()._check_positions_type_multiPos(positions=positions)
        # dim check
        if (all([len(pos) == cls.nDim for pos in positions])):
            return positions
        else:
            raise Exception("Dimensionality is not correct for positions! " + str(positions))


"""
MultiState Potentials
"""


class _potentialNDMultiState(_potentialNDCls):
    nStates: int = 2
    states: _potentialNDCls

    def __init__(self, nDim: int, nStates: int):
        super().__init__(nDim=nDim)
        self.nStates = nStates

    def _check_positions_type_singlePos(self, position: Union[Iterable[Number], Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param position: here the positions that shall be evaluated by the potential should be given.
        :type position:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        # array
        # print(position)
        if (isinstance(position, Number)):
            if (self.nDim == 1):
                return np.array([position for state in range(self.nStates)])
            elif (self.nDim == -1):
                return np.array([[position, ] for state in range(self.nStates)])
        elif (isinstance(position, Iterable) and isinstance(position, Sized)):
            if ((len(position) == self.nDim or self.nDim == -1) and all([isinstance(dim, Number) for dim in position])):
                return np.array([position for state in range(self.nStates)])
            elif (len(position) == self.nStates):
                if (all([isinstance(state, Iterable) and (len(state) == self.nDim or self.nDim == -1) for state in
                         position]) and
                        all([[isinstance(dim, Number) for dim in state] for state in position])):
                    return np.array(position)
                if (self.nDim == 1 and all([isinstance(dim, Number) for dim in position])):
                    return np.array([[p] for p in position])
            elif (len(position) == 1 and isinstance(position[0], Number)):
                return np.array([position[0] for state in range(self.nStates)])
            else:
                raise Exception(
                    "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))
        else:
            raise Exception(
                "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))

    def _check_positions_type_multiPos(self, positions: Union[
        Iterable[Iterable[Number]], Iterable[Number], Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  Union[Iterable[Number], Number]
        :return: returns the evaluated potential values
        :return type: Iterable[Number]
        """
        # array
        if (isinstance(positions, Number) and (self.nDim == 1 or self.nDim == -1)):
            return np.array([[positions, ] for state in range(self.nStates)])
        elif (isinstance(positions, Iterable) and isinstance(positions, Sized)):
            if ((len(positions) == self.nDim or self.nDim == -1) and all(
                    [isinstance(dim, Number) for dim in positions])):
                return np.array([positions for state in range(self.nStates)])
            elif ((1 == self.nDim or self.nDim == -1) and all([isinstance(dim, Number) for dim in positions])):
                return np.array([[p for state in range(self.nStates)] for p in positions])
            elif (len(positions) == self.nStates and all(
                    [(len(state) == self.nDim or self.nDim == -1) for state in positions]) and
                  all([[isinstance(dim, Number) for dim in state] for state in positions])):
                return np.array(positions)
            else:
                raise Exception(
                    "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))
        else:
            raise Exception(
                "Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))


class _perturbedPotentialNDCls(_potentialNDMultiState):
    """
        .. autoclass:: perturbedPotentialCls
    """
    nDim: int = -1
    nStates: int
    lam: Number

    def __init__(self, state_potentials: Iterable[_potentialNDCls], lam: Number = 0.0):
        '''
        Initializes a potential of the form V = 0.5 * (1 + alpha * lam) * fc * (pos - gamma * lam) ** 2
        :param fc: force constant
        :param alpha: perturbation parameter for width of harmonic oscillator
        :param gamma: perturbation parameter for position of harmonic oscillator
        '''
        self.states = state_potentials
        self.nStates = len(state_potentials)
        if (all([V.nDim == state_potentials[0].nDim for V in state_potentials])):
            self.nDim = state_potentials[0].nDim

        super().__init__(nDim=self.nDim, nStates=self.nStates)
        self.lam = lam
        self._calculate_dhdlam = self._calculate_dhdlam_multiPos

    def set_lam(self, lam: Number):
        self.lam = lam

    def dhdlam(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates derivative with respect to lambda value
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dlan
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdlam(positions)

    def _calculate_dhdlam(self, positions: Iterable[Number], lam: Number = 1.0):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdlam_singlePos(self, position: Number) -> (np.array or Number):
        raise Exception("Please implement this function!")

    def _calculate_dhdlam_multiPos(self, positions: (Iterable[Number] or Number)) -> (np.array or Number):
        return np.array(list(map(self._calculate_dhdlam_singlePos, positions)))


"""
NEW SYMPY IMPLEMENT
"""
import sympy as sp


def notImplementedERR():
    raise NotImplementedError("This function needs to be implemented in sympy")


"""
SYMPY POTENTIALS
"""


class _potentialNDClsSymPY(_potentialCls):
    '''
    potential base class
    @nullState
    @Strategy Pattern
    '''
    name: str = "Unknown"
    nDim: sp.symbol = sp.symbols("nDim")
    nStates: int = 1
    _no_Type_check: bool = False
    _singlePos_mode: bool = False

    position: sp.Symbol = None
    V_orig: sp.Function
    dVdpos_orig: sp.Function

    V: sp.Function = notImplementedERR
    dVdpos = notImplementedERR
    constants: dict = {}

    def __init__(self, nDim: int = -1):
        self.constants.update({self.nDim: nDim})
        # needed for multi dim functions to be generated dynamically
        self._initialize_functions()
        # apply potential simplification and update calc functions
        self._update_functions()

    def __name__(self) -> str:
        return str(self.name)

    def __str__(self) -> str:
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.nStates) + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
        msg += "\n\tFunctional:\n "
        msg += "\t\tV:\t" + str(self.V_orig) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos_orig) + "\n"
        msg += "\n\tSimplified Function\n"
        msg += "\t\tV:\t" + str(self.V) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos) + "\n"
        msg += "\n\tConstants: \n\t\t" + "\n\t\t".join(
            [str(keys) + ": " + "\n".join(["\t\t\t" + v for v in str(values).split("\n")]) for keys, values in
             self.constants.items()]) + "\n"
        msg += "\n"
        return msg

    """
        private
    """

    def _initialize_functions(self):
        notImplementedERR()

    def _update_functions(self):
        self.V = self.V_orig.subs(self.constants)

        self.dVdpos_orig = sp.diff(self.V_orig, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")

    def _set_type_check(self):
        """
        ..autofunction :: _set_type_check
            This function is setting the default potential Value, to allow secure execution in small code snippets
        :return:  -
        """
        pass

    def _check_positions_type(self):
        pass

    def set_simulation_mode(self, simulation: bool = True):
        pass

    """
        public
    """

    def ene(self, positions):
        return np.squeeze(self._calculate_energies(*np.hsplit(positions, self.constants[self.nDim])))

    def dvdpos(self, positions):
        return np.squeeze(self._calculate_dVdpos(*np.hsplit(positions, self.constants[self.nDim])))


class _potential1DClsSymPY(_potentialNDClsSymPY):
    def __init__(self):
        super().__init__(nDim=1)

    def _initialize_functions(self):
        pass

    def ene(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates energy of particle
        :param pos: position on 1D potential energy surface
        :return: energy
        '''
        return np.squeeze(self._calculate_energies(np.squeeze(positions)))

    def dvdpos(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates derivative with respect to position
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        return np.squeeze(self._calculate_dVdpos(np.squeeze(positions)))


class _potential2DClsSymPY(_potentialNDClsSymPY):
    def __init__(self):
        super().__init__(nDim=2)


class _potential1DClsSymPYPerturbed(_potential1DClsSymPY):
    nStates: int = 1

    lam = sp.symbols(u"λ")

    V_orig = sp.Function
    dVdlam_orig: sp.Function

    statePotentials: Dict[sp.Function, sp.Function]

    dVdlam = notImplementedERR

    def __init__(self):
        self.nStates = 1
        self.nDim = 1
        super().__init__()
        self.nStates = 1
        self.nDim = 1  # check this

    def __str__(self) -> str:
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.nStates) + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
        msg += "\n\tFunctional:\n "
        msg += "\t\tCoupling:\t" + str(self.Coupling) + "\n"
        msg += "\t\tV:\t" + str(self.V_orig) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos_orig) + "\n"
        msg += "\t\tdVdlam:\t" + str(self.dVdlam_orig) + "\n"
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
        # workaround for state definition.
        self.V_orig = self.Coupling
        for key, f in self.statePotentials.items():
            self.V_orig = self.V_orig.replace(key, f._calculate_energies)

        super()._update_functions()

        self.dVdlam_orig = sp.diff(self.V_orig, self.lam)
        self.dVdlam = self.dVdlam_orig.subs(self.constants)
        self._calculate_dVdlam = sp.lambdify(self.position, self.dVdlam, "numpy")

    """
        public
    """

    def set_lam(self, lam: float):
        self.constants.update({self.lam: lam})
        self._update_functions()

    def dvdlam(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates derivative with respect to lambda
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        return np.squeeze(self._calculate_dVdlam(np.squeeze(positions)))

