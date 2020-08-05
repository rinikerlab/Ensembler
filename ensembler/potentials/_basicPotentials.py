from numbers import Number
from typing import Iterable, Sized, Union, Dict

import numpy as np
import sympy as sp

from ensembler.util.basic_class import super_baseClass, notImplementedERR


# from concurrent.futures.thread import ThreadPoolExecutor

class _potentialCls(super_baseClass):
    """
    potential base class
    @nullState
    @Strategy Pattern
    """
    nDim: sp.Symbol = sp.symbols("nDims")
    nStates: sp.Symbol = sp.symbols("nStates")
    # threads: int = 1

    # hidden attributes
    __constants: Dict[sp.Symbol, Union[Number, Iterable]] = {}

    def __init__(self):
        self.name = self.__class__.__name__

    """
        Non-Class Attributes
    """

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
    potential base class
    @nullState
    @Strategy Pattern
    '''

    # generated during construction:
    position: sp.Symbol("r")

    V_functional: sp.Function = notImplementedERR
    dVdpos_functional: sp.Function = notImplementedERR

    V: sp.Function = notImplementedERR
    dVdpos = notImplementedERR

    def __init__(self, nDim: int = -1, nStates: int = 1):
        super().__init__()

        self.constants.update({self.nDim: nDim, self.nStates: nStates})
        # needed for multi dim functions to be generated dynamically
        self._initialize_functions()
        # apply potential simplification and update calc functions
        self._update_functions()

    def __str__(self) -> str:
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.constants[self.nStates]) + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
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

    """
        private
    """

    def _initialize_functions(self):
        notImplementedERR()

    def __setstate__(self, state):
        """
        Setting up after pickling.
        """
        self.__dict__ = state
        self._initialize_functions()
        self._update_functions()

    def _update_functions(self):
        self.V = self.V_functional.subs(self.constants).expand()

        self.dVdpos_functional = sp.diff(self.V_functional, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")

    """
        public
    """
    _calculate_energies = lambda x: notImplementedERR()
    _calculate_dVdpos = lambda x: notImplementedERR()

    def ene(self, positions: Union[Number, Sized]) -> Union[Number, Sized]:
        return np.squeeze(self._calculate_energies(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDim])))

    def force(self, positions: Union[Number, Sized]) -> Union[Number, Sized]:
        return np.squeeze(self._calculate_dVdpos(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDim])))

    # just alternative name
    def dvdpos(self, positions: Union[Number, Sized]) -> Union[Number, Sized]:
        return self.force(positions)


class _potential1DCls(_potentialNDCls):

    def __init__(self, nStates: int = 1):
        super().__init__(nDim=1, nStates=nStates)

    def _initialize_functions(self):
        pass

    def ene(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates energy of particle
        :param pos: position on 1D potential energy surface
        :return: energy
        '''
        return np.squeeze(self._calculate_energies(np.array(positions)))

    def force(self, positions: (Iterable[Number] or Number)) -> (Iterable[Number] or Number):
        '''
        calculates derivative with respect to position
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        return np.squeeze(self._calculate_dVdpos(np.squeeze(np.array(positions))))


class _potential2DCls(_potentialNDCls):
    def __init__(self, nStates: int = 1):
        super().__init__(nDim=2, nStates=nStates)


class _potential1DClsPerturbed(_potential1DCls):
    coupling: sp.Function = notImplementedERR

    lam = sp.symbols(u"λ")
    statePotentials: Dict[sp.Function, sp.Function]

    dVdlam_functional: sp.Function
    dVdlam = notImplementedERR

    def __init__(self, nStates: int = 1):
        self.constants.update({self.nDim: 1, self.nStates: nStates})

        super().__init__(nStates=nStates)

    def __str__(self) -> str:
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.constants[self.nStates]) + "\n"
        msg += "\tDimensions: " + str(self.nDim) + "\n"
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
        self.V_functional = self.coupling

        super()._update_functions()

        self.dVdlam_functional = sp.diff(self.V_functional, self.lam)
        self.dVdlam = self.dVdlam_functional.subs(self.constants)
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
