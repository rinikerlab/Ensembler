"""
Module: Potential
    This module shall be used to implement subclasses of Potentials that formulate a potential as an Function with N-Dimensions.
     This module contains all available potentials.
"""

import numpy as np
import sympy as sp
from ensembler.util import ensemblerTypes as t
from ensembler.util.ensemblerTypes import Number, Union, Iterable

# Base Classes
from ensembler.potentials._basicPotentials import _potentialNDCls


class harmonicOscillatorPotential(_potentialNDCls):
    """
    ND  harmonic oscillator potential
    """

    name: str = "harmonicOscilator"
    nDimensions: int = sp.symbols("nDimensions")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    r_shift: sp.Matrix = sp.Matrix([sp.symbols("r_shift")])
    Voff: sp.Matrix = sp.Matrix([sp.symbols("V_off")])
    k: sp.Matrix = sp.Matrix([sp.symbols("k")])

    V_dim = 0.5 * k * (position - r_shift) ** 2 + Voff

    i = sp.Symbol("i")
    V_functional = sp.Sum(V_dim[i, 0], (i, 0, nDimensions))

    def __init__(
        self,
        k: np.array = np.array([1.0, 1.0, 1.0]),
        r_shift: np.array = np.array([0.0, 0.0, 0.0]),
        Voff: np.array = np.array([0.0, 0.0, 0.0]),
        nDimensions: int = 3,
    ):
        """
            __init__
                Constructs an harmonic Oscillator with an on runtime defined dimensionality.

        Parameters
        ----------
        k: List[float], optional
            force constants, as many as nDim, defaults to  [1.0, 1.0, 1.0]
        x_shift: List[float], optional
            shift of the minimum in the x Axis, as many as nDim, defaults to [0.0, 0.0, 0.0]
        y_shift: List[float], optional
            shift on the y Axis, as many as nDim, defaults to [0.0, 0.0, 0.0]
        nDim
            dimensionality of the harmoic oscillator object. default: 3
        """
        self.constants = {self.nDimensions: nDimensions}
        self.constants.update({"k_" + str(j): k[j] for j in range(self.constants[self.nDimensions])})
        self.constants.update({"r_shift" + str(j): r_shift[j] for j in range(self.constants[self.nDimensions])})
        self.constants.update({"V_off_" + str(j): Voff[j] for j in range(self.constants[self.nDimensions])})

        super().__init__(nDimensions=nDimensions)

    def _initialize_functions(self):
        """
        Build up the nDimensionssymbolic definitions
        """
        # Parameters
        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])
        self.r_shift = sp.Matrix([sp.symbols("r_shift" + str(i)) for i in range(nDimensions)])
        self.V_off = sp.Matrix([sp.symbols("V_off_" + str(i)) for i in range(nDimensions)])
        self.k = sp.Matrix([sp.symbols("k_" + str(i)) for i in range(nDimensions)])
        # Function
        self.V_dim = 0.5 * sp.matrix_multiply_elementwise(
            self.k, ((self.position - self.r_shift).applyfunc(lambda x: x**2))
        )  # +self.Voff
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDimensions - 1))


class envelopedPotential(_potentialNDCls):
    """
    This implementation of exponential Coupling for EDS is a more numeric robust and variable implementation, it allows N states.
    Therefore the computation of energies and the deviation is not symbolic.

    Here N-states are coupled by the log-sum-exp resulting in a new reference state $V_R$,

    $V_R = -1/{\beta} * \ln(\sum_i^Ne^(-\beta*s*(V_i-E^R_i)))$

    This potential coupling is for example used in EDS.
    """

    name = "Enveloping Potential"

    T, kb, position = sp.symbols("T kb r")
    beta = 1 / (kb * T)

    Vis = sp.Matrix(["V_i"])
    Eoffis = sp.Matrix(["Eoff_i"])
    sis = sp.Matrix(["s_i"])
    i, nStates = sp.symbols("i N")
    V_functional = -1 / (beta * sis[0, 0]) * sp.log(sp.Sum(sp.exp(-beta * sis[i, 0] * (Vis[i, 0] - Eoffis[i, 0])), (i, 0, nStates)))

    def __init__(
        self,
        V_is: t.List[_potentialNDCls] = (
            harmonicOscillatorPotential(nDimensions=2),
            harmonicOscillatorPotential(r_shift=[3, 3], nDimensions=2),
        ),
        s: float = 1.0,
        eoff: t.List[float] = None,
        T: float = 1,
        kb: float = 1,
    ):
        """
            __init__
                This function constructs a enveloped potential, enveloping all given states.

        Parameters
        ----------
        V_is: List[_potential1DCls], optional
            The states(potential classes) to be enveloped (default: [harmonicOscillatorPotential(), harmonicOscillatorPotential(x_shift=3)])

        s: float, optional
            the smoothing parameter, lowering the barriers between the states
        eoff: List[float], optional
            the energy offsets of the individual states in the reference potential. These can be used to allow a more uniform sampling. (default: seta ll to 0)
        T: float, optional
            the temperature of the reference state (default: 1 = T)
        kb: float, optional
            the boltzman constant (default: 1 = kb)

        """
        self.constants = {self.T: T, self.kb: kb}
        nStates = len(V_is)
        self._Eoff_i = [0 for x in range(nStates)]
        self._s = [0 for x in range(nStates)]
        self._V_is = [0 for x in range(nStates)]

        # for calculate implementations
        self.V_is = V_is
        self.s_i = s
        self.Eoff_i = eoff

        super().__init__(nDimensions=V_is[0].constants[V_is[0].nDimensions], nStates=len(V_is))

    def _initialize_functions(self):
        """
        build the symbolic functionality.
        """
        # for sympy Sympy Updates - Check!:
        self.statePotentials = {"state_" + str(j): self.V_is[j] for j in range(self.constants[self.nStates])}
        Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
        sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
        keys = zip(sorted(self.statePotentials.keys()), sorted(Eoffis.keys()), sorted(sis.keys()))

        self.states = sp.Matrix([sp.symbols(l) * (sp.symbols(j) - sp.symbols(k)) for j, k, l in keys])
        self.constants.update({**{state: value.V for state, value in self.statePotentials.items()}, **Eoffis, **sis})

        self.V_functional = (
            -1 / (self.beta * self.sis[0, 0]) * sp.log(sp.Sum(sp.exp(-self.beta * self.states[self.i, 0]), (self.i, 0, self.nStates - 1)))
        )
        self._update_functions()

        # also make sure that states are up to work:
        [V._update_functions() for V in self.V_is]

        if all([self.s_i[0] == s for s in self.s_i[1:]]):
            self.ene = self._calculate_energies_singlePos_overwrite_oneS
        else:
            self.ene = self._calculate_energies_singlePos_overwrite_multiS
        self.force = self._calculate_dvdpos_singlePos_overwrite

    @property
    def V_is(self) -> t.List[_potentialNDCls]:
        """
        V_is are the state potential classes enveloped by the reference state.

        Returns
        -------
        V_is: t.List[_potential1DCls]
        """
        return self._V_is

    @V_is.setter
    def V_is(self, V_is: t.List[_potentialNDCls]):
        if isinstance(V_is, Iterable) and all([isinstance(Vi, _potentialNDCls) for Vi in V_is]):
            self._V_is = V_is
            self.constants.update({self.nStates: len(V_is)})
        else:
            raise IOError("Please give the enveloped potential for V_is only 1D-Potential classes in a list.")

    def set_Eoff(self, Eoff: Union[Number, Iterable[Number]]):
        """
        This function is setting the Energy offsets of the states enveloped by the reference state.
        Parameters
        ----------
        Eoff: Union[Number, Iterable[Number]]
        """
        self.Eoff_i = Eoff

    @property
    def Eoff(self) -> t.List[Number]:
        """
        The Energy offsets are used to bias the single states in the reference potential by a constant offset.
        Therefore each state of the enveloping potential has its own energy offset.

        Returns
        -------
        Eoff:t.List[Number]

        """
        return self.Eoff_i

    @Eoff.setter
    def Eoff(self, Eoff: Union[Number, Iterable[Number], None]):
        self.Eoff_i = Eoff

    @property
    def Eoff_i(self) -> t.List[Number]:
        """
        The Energy offsets are used to bias the single states in the reference potential by a constant offset.
        Therefore each state of the enveloping potential has its own energy offset.

        Returns
        -------
        Eoff:t.List[Number]

        """
        return self._Eoff_i

    @Eoff_i.setter
    def Eoff_i(self, Eoff: Union[Number, Iterable[Number], None]):
        if isinstance(Eoff, type(None)):
            self._Eoff_i = [0.0 for state in range(self.constants[self.nStates])]
            Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**Eoffis})
        elif len(Eoff) == self.constants[self.nStates]:
            self._Eoff_i = Eoff
            Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**Eoffis})
        else:
            raise IOError(
                "Energy offset Vector and state potentials don't have the same length!\n states in Eoff "
                + str(len(Eoff))
                + "\t states in Vi"
                + str(len(self.V_is))
            )

    def set_s(self, s: Union[Number, Iterable[Number]]):
        """
            set_s
            is a function used to set an smoothing parameter.
        Parameters
        ----------
        s:Union[Number, Iterable[Number]]

        Returns
        -------

        """
        self.s_i = s

    @property
    def s(self) -> t.List[Number]:
        return self.s_i

    @s.setter
    def s(self, s: Union[Number, Iterable[Number]]):
        self.s_i = s

    @property
    def s_i(self) -> t.List[Number]:
        return self._s

    @s_i.setter
    def s_i(self, s: Union[Number, Iterable[Number]]):
        if isinstance(s, Number):
            self._s = [s for x in range(self.constants[self.nStates])]
            sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**sis})
        elif len(s) == self.constants[self.nStates]:
            self._s = s
            sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**sis})
        else:
            raise IOError(
                "s Vector/Number and state potentials don't have the same length!\n states in s "
                + str(len(s))
                + "\t states in Vi"
                + str(len(self.V_is))
            )
        self._update_functions()

    def _calculate_energies_singlePos_overwrite_multiS(self, positions) -> np.array:
        sum_prefactors, _ = self._logsumexp_calc_gromos(positions)
        beta = self.constants[self.T] * self.constants[self.kb]  # kT - *self.constants[self.T]
        Vr = (-1 / (beta)) * sum_prefactors
        return np.squeeze(Vr)

    def _calculate_energies_singlePos_overwrite_oneS(self, positions) -> np.array:
        sum_prefactors, _ = self._logsumexp_calc(positions)
        beta = self.constants[self.T] * self.constants[self.kb]
        Vr = (-1 / (beta * self.s_i[0])) * sum_prefactors
        return np.squeeze(Vr)

    def _calculate_dvdpos_singlePos_overwrite(self, positions: (t.Iterable[float])) -> np.array:
        """
        Parameters
        ----------
        positions

        Returns
        -------

        """
        positions = np.array(positions, ndmin=2)
        # print("Pos: ", position)

        V_R_part, V_Is_ene = self._logsumexp_calc_gromos(positions)
        V_R_part = np.array(V_R_part, ndmin=2).T
        # print("V_R_part: ", V_R_part.shape, V_R_part)
        # print("V_I_ene: ",V_Is_ene.shape, V_Is_ene)
        V_Is_dhdpos = np.array([-statePot.force(positions) for statePot in self.V_is], ndmin=1).T
        # print("V_I_force: ",V_Is_dhdpos.shape, V_Is_dhdpos)

        adapt = np.concatenate([V_R_part for s in range(self.constants[self.nStates])], axis=1)
        # print("ADAPT: ",adapt.shape, adapt)
        scaling = np.exp(V_Is_ene - adapt)
        # print("scaling: ", scaling.shape, scaling)
        dVdpos_state = np.multiply(scaling, V_Is_dhdpos)  # np.array([(ene/V_R_part) * force for ene, force in zip(V_Is_ene, V_Is_dhdpos)])
        # print("state_contributions: ",dVdpos_state.shape, dVdpos_state)
        dVdpos = np.sum(dVdpos_state, axis=1)
        # print("forces: ",dVdpos.shape, dVdpos)

        return np.squeeze(dVdpos)

    def _logsumexp_calc(self, position):
        prefactors = []
        beta = self.constants[self.T] * self.constants[self.kb]
        for state in range(self.constants[self.nStates]):
            prefactor = np.array(-beta * self.s_i[state] * (self.V_is[state].ene(position) - self.Eoff_i[state]), ndmin=1).T
            prefactors.append(prefactor)
        prefactors = np.array(prefactors, ndmin=2).T

        from scipy.special import logsumexp

        # print("Prefactors", prefactors)
        sum_prefactors = logsumexp(prefactors, axis=1)
        # print("logexpsum: ", np.squeeze(sum_prefactors))

        return np.squeeze(sum_prefactors), np.array(prefactors, ndmin=2).T

    def _logsumexp_calc_gromos(self, position):
        """
        code from gromos:

        Parameters
        ----------
        position

        Returns
        -------

        """
        prefactors = []
        beta = self.constants[self.T] * self.constants[self.kb]  # kT - *self.constants[self.T]
        partA = np.array(-beta * self.s_i[0] * (self.V_is[0].ene(position) - self.Eoff_i[0]), ndmin=1)
        partB = np.array(-beta * self.s_i[1] * (self.V_is[1].ene(position) - self.Eoff_i[1]), ndmin=1)

        partAB = np.array([partA, partB]).T
        log_prefac = 1 + np.exp(np.min(partAB, axis=1) - np.max(partAB, axis=1))
        sum_prefactors = np.max(partAB, axis=1) + np.log(log_prefac)

        prefactors.append(partA)
        prefactors.append(partB)

        # more than two states!
        for state in range(2, self.constants[self.nStates]):
            partN = np.array(-beta * self.s_i[state] * (self.V_is[state].ene(position) - self.Eoff_i[state]), ndmin=1)
            prefactors.append(partN)
            sum_prefactors = np.max([sum_prefactors, partN], axis=1) + np.log(
                1 + np.exp(np.min([sum_prefactors, partN], axis=1) - np.max([sum_prefactors, partN], axis=1))
            )
            # print("prefactors: ", sum_prefactors)
        return sum_prefactors, np.array(prefactors, ndmin=2).T


class lambdaEDSPotential(envelopedPotential):
    """
    This implementation of exponential Coupling combined with linear compling is called $\lambda$-EDS the implementation of function is more numerical robust to the hybrid coupling class.


    Here two-states are coupled by the log-sum-exp and weighted by lambda resulting in a new reference state $V_R$,

    $V_R = -\frac{1}{\beta*s} * \ln(\lambda * e^(-\beta*s*(V_A-E^R_A)) + (1-\lambda)*e^(-\beta*s*(V_B-E^R_B)))$

    This potential coupling is for example used in $\lambda$-EDS.
    """

    name: str = "lambda enveloped Potential"

    T, kb, position = sp.symbols("T kb r")
    beta = 1 / (kb * T)

    Vis = sp.Matrix(["V_i"])
    Eoffis = sp.Matrix(["Eoff_i"])
    sis = sp.Matrix(["s_i"])
    lamis = sp.Matrix(["λ"])

    i, nStates = sp.symbols("i N")
    V_functional = (
        -1 / (beta * sis[0, 0]) * sp.log(sp.Sum(lamis[i, 0] * sp.exp(-beta * sis[i, 0] * (Vis[i, 0] - Eoffis[i, 0])), (i, 0, nStates)))
    )

    def __init__(
        self,
        V_is: t.List[_potentialNDCls] = (
            harmonicOscillatorPotential(nDimensions=2),
            harmonicOscillatorPotential(r_shift=[3, 3], nDimensions=2),
        ),
        lam: Number = 0.5,
        s: float = 1.0,
        eoff: t.List[float] = None,
        T: float = 1,
        kb: float = 1,
    ):

        nStates = len(V_is)
        self.constants = {self.nStates: nStates}
        self._Eoff_i = [0 for x in range(nStates)]
        self._s = [0 for x in range(nStates)]
        self._V_is = [0 for x in range(nStates)]
        self._lam_i = [0 for x in range(nStates)]
        self.lam_i = lam

        super().__init__(V_is=V_is, s=s, eoff=eoff, T=T, kb=kb)

    def _initialize_functions(self):
        # for sympy Sympy Updates - Check!:
        self.statePotentials = {"state_" + str(j): self.V_is[j] for j in range(self.constants[self.nStates])}
        Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
        sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
        lamis = {"lam_" + str(i): self.lam_i[i] for i in range(self.constants[self.nStates])}
        keys = zip(sorted(self.statePotentials.keys()), sorted(Eoffis.keys()), sorted(sis.keys()))

        self.states = sp.Matrix([sp.symbols(l) * (sp.symbols(j) - sp.symbols(k)) for j, k, l in keys])
        self.constants.update({**{state: value.V for state, value in self.statePotentials.items()}, **Eoffis, **sis, **lamis})
        inner_log = sp.Sum(
            sp.Matrix(list(lamis.keys()))[self.i, 0] * sp.exp(-self.beta * self.states[self.i, 0]), (self.i, 0, self.nStates - 1)
        )
        self.V_functional = -1 / (self.beta * self.sis[0, 0]) * sp.log(inner_log)
        self._update_functions()

        # also make sure that states are up to work:
        [V._update_functions() for V in self.V_is]

        self.ene = self._calculate_energies_singlePos_overwrite
        self.force = self._calculate_dvdpos_singlePos_overwrite

    def set_lam(self, lam: Union[Number, Iterable[Number]]):
        self.lam_i = lam

    @property
    def lam(self) -> t.List[Number]:
        return self.lam_i

    @lam.setter
    def lam(self, lam: Union[Number, Iterable[Number]]):
        self.lam_i = lam

    @property
    def lam_i(self) -> t.List[Number]:
        return self._lam_i

    @lam_i.setter
    def lam_i(self, lam: Union[Number, Iterable[Number]]):
        if isinstance(lam, Number) and self.constants[self.nStates] == 2:
            self._lam_i = np.array([lam] + [1 - lam for x in range(1, self.constants[self.nStates])], ndmin=1)
            lamis = {"lam_" + str(i): self.lam_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**lamis})
        elif isinstance(lam, Number):
            self._lam_i = np.array([1 / self.constants[self.nStates] for x in range(self.constants[self.nStates])], ndmin=1)
            lamis = {"lam_" + str(i): self.lam_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**lamis})
        elif len(lam) == self.constants[self.nStates]:
            raise NotImplementedError("Currently Only one lam runs supported!")
            # self._lam_i = np.array(lam, ndmin=1)
            # self.constants.update({self.lamis: self._lam_i})
        else:
            raise IOError(
                "s Vector/Number and state potentials don't have the same length!\n states in s "
                + str(lam)
                + "\t states in Vi"
                + str(len(self.V_is))
            )

    def _calculate_energies_singlePos_overwrite(self, position) -> np.array:
        # print("Positions: ",position)
        # print("s_i: ",self.s_i)
        sum_prefactors, _ = self._logsumexp_calc(position)
        beta = self.constants[self.T] * self.constants[self.kb]  # kT - *self.constants[self.T]
        Vr = (-1 / (beta * self.s_i[0])) * sum_prefactors
        # print("finalVR", Vr)
        return np.squeeze(Vr)

    def _calculate_dvdpos_singlePos_overwrite(self, positions: (t.Iterable[float])) -> np.array:
        positions = np.array(positions, ndmin=2)
        # print("Pos: ", position)
        V_R_part, V_Is_ene = self._logsumexp_calc(positions)
        # print("V_I_ene: ",V_Is_ene.shape, V_Is_ene)
        V_R_part = np.array(V_R_part, ndmin=2).T
        # print("V_R_part: ", V_R_part.shape, V_R_part)
        V_Is_dhdpos = np.array([-statePot.force(positions) for statePot in self.V_is], ndmin=1).T
        # print("V_I_force: ",V_Is_dhdpos.shape, V_Is_dhdpos)
        adapt = np.concatenate([V_R_part for s in range(self.constants[self.nStates])], axis=1).T
        # print("ADAPT: ",adapt.shape, adapt)
        # print(self.lam_i)
        scaling = (np.array(self.lam_i, ndmin=2).T * (np.exp(V_Is_ene - adapt))).T

        # print("scaling: ", scaling.shape, scaling)
        dVdpos_state = scaling * V_Is_dhdpos
        # print("state_contributions: ",dVdpos_state.shape, dVdpos_state)
        dVdpos = np.sum(dVdpos_state, axis=1)
        # print("forces: ",dVdpos.shape, dVdpos)

        return np.squeeze(dVdpos)

    def _logsumexp_calc(self, position):
        prefactors = []
        beta = self.constants[self.T] * self.constants[self.kb]
        for state in range(self.constants[self.nStates]):
            prefactor = np.array(-beta * self.s_i[state] * (self.V_is[state].ene(position) - self.Eoff_i[state]), ndmin=1).T
            prefactors.append(prefactor)
        prefactors = np.array(prefactors, ndmin=2).T

        from scipy.special import logsumexp

        sum_prefactors = logsumexp(prefactors, axis=1, b=self.lam)

        return np.squeeze(sum_prefactors), np.array(prefactors, ndmin=2).T


class sumPotentials(_potentialNDCls):
    """
    Adds n different potentials.
     For adding up wavepotentials, we recommend using the addedwavePotential class.
    """

    name: str = "Summed Potential"

    position = sp.symbols("r")
    potentials: sp.Matrix = sp.Matrix([sp.symbols("V_x")])

    nPotentials = sp.symbols("N")
    i = sp.symbols("i", cls=sp.Idx)

    V_functional = sp.Sum(potentials[i, 0], (i, 0, nPotentials))

    def __init__(
        self,
        potentials: t.List[_potentialNDCls] = (
            harmonicOscillatorPotential(),
            harmonicOscillatorPotential(r_shift=[1, 1, 1], nDimensions=3),
        ),
    ):
        """
        __init__
            This is the Constructor of an summed Potentials

        Parameters
        ----------
        potentials: List[_potential2DCls], optional
            it uses the 2D potential class to generate its potential,
            default to (wavePotential(), wavePotential(multiplicity=[3, 3]))
        """
        if all([potentials[0].constants[V.nDimensions] == V.constants[V.nDimensions] for V in potentials]):
            nDim = potentials[0].constants[potentials[0].nDimensions]
        else:
            raise ValueError(
                "The potentials don't share the same dimensionality!\n\t" + str([V.constants[V.nDimensions] for V in potentials])
            )

        self.constants = {self.nPotentials: len(potentials)}
        self.constants.update({"V_" + str(i): potentials[i].V for i in range(len(potentials))})

        super().__init__(nDimensions=nDim)

    def _initialize_functions(self):
        """
        _initialize_functions
            converts the symbolic mathematics of sympy to a matrix representation that is compatible
            with multi-dimentionality.
        """
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(self.constants[self.nDimensions])])
        self.potentials = sp.Matrix([sp.symbols("V_" + str(i)) for i in range(self.constants[self.nPotentials])])
        # Function
        self.V_functional = sp.Sum(self.potentials[self.i, 0], (self.i, 0, self.nPotentials - 1))

    def __str__(self) -> str:
        msg = self.__name__() + "\n"
        msg += "\tStates: " + str(self.constants[self.nStates]) + "\n"
        msg += "\tDimensions: " + str(self.nDimensions) + "\n"
        msg += "\n\tFunctional:\n "
        msg += "\t\tV:\t" + str(self.V_functional) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos_functional) + "\n"
        msg += "\n\tSimplified Function\n"
        msg += "\t\tV:\t" + str(self.V) + "\n"
        msg += "\t\tdVdpos:\t" + str(self.dVdpos) + "\n"
        msg += "\n"
        return msg

    # OVERRIDE
    def _update_functions(self):
        """
        _update_functions
            calculates the current energy and derivative of the energy
        """
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos
