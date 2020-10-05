"""
Module: Potential
This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import typing as t

import numpy as np
import scipy.constants as const
import sympy as sp

from ensembler.potentials._basicPotentials import _potential1DCls, _potential1DClsPerturbed
from ensembler.util.ensemblerTypes import Union, Number, Iterable, systemCls

"""
    SIMPLE POTENTIALS
"""


class harmonicOscillatorPotential(_potential1DCls):
    """
        Implementation of an 1D  harmonic oscillator potential following hooke's law
    """
    name: str = "Harmonic Oscillator"
    k, x_shift, position, y_shift = sp.symbols("k r_0 r Voffset")
    V_functional = 0.5 * k * (position - x_shift) ** 2 + y_shift

    def __init__(self, k: float = 1.0, x_shift: float = 0.0, y_shift: float = 0.0):
        """
        __init__
            This is the Constructor of the 1D harmonic oscillator

        Parameters
        ----------
        k: float, optional
            force constant, defaults to 1.0
        x_shift: float, optional
            shift of the minimum in the x Axis, defaults to 0.0
        y_shift: float, optional
            shift on the y Axis, defaults to 0.0
        """

        self.constants = {self.k: k, self.x_shift: x_shift, self.y_shift: y_shift}
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


class wavePotential(_potential1DCls):
    """
       Simple 1D wave potential consisting of a cosine function with given multiplicity, that can be shifted and elongated
       """
    name: str = "Wave Potential"
    amplitude, phase_shift, position, y_shift, multiplicity = sp.symbols("A w r Voff m")
    V_functional = amplitude * sp.cos(multiplicity * (position + phase_shift)) + y_shift

    def __init__(self, amplitude: float = 1.0, multiplicity: float = 1.0, phase_shift: float = 0.0,
                 y_shift: float = 0.0, radians: bool = False):
        """
        __init__
            This is the Constructor of the 1D wave potential function
        Parameters
        ----------
        amplitude: float, optional
            absolute min and max of the potential, defaults to 1.0
        multiplicity: float, optional
            amount of minima in one phase, defaults to 1.0
        phase_shift: float, optional
            shift of the potential on the x Axis, defaults to 0.0
        y_offset: float, optional
            shift on the y Axis, defaults to 0.0
        radians: bool, optional
            in radians or degrees, defaults to False
        """

        self.constants = {self.amplitude: amplitude, self.multiplicity: multiplicity, self.phase_shift: phase_shift,
                          self.y_shift: y_shift}
        super().__init__()
        self.set_radians(radians)

    # OVERRIDE
    def _update_functions(self):
        """
        _update_functions
            calculates the current energy and derivative of the energy
        """
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos

    def set_degrees(self, degrees: bool = True):
        """
        Sets output to either degrees or radians

        Parameters
        ----------
        degrees: bool, optional,
            if True, output will be given in degrees, otherwise in radians, default: True
        """
        self.radians = not degrees
        if (degrees):
            self._calculate_energies = lambda positions: self.tmp_Vfunc(np.deg2rad(positions))
            self._calculate_dVdpos = lambda positions: self.tmp_dVdpfunc(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        """
        Sets output to either degrees or radians

        Parameters
        ----------
        radians: bool, optional,
            if True, output will be given in radians, otherwise in degree, default: True
        """
        self.radians = radians
        if (radians):
            self._calculate_energies = self.tmp_Vfunc
            self._calculate_dVdpos = self.tmp_dVdpfunc
        else:
            self.set_degrees(degrees=not radians)


class coulombPotential(_potential1DCls):
    """
    Coulomb potential representing the pairwise electrostatic interaction of two charged particles
    """
    name = "Coulomb Potential"
    charge1, charge2, position, electric_permetivity = sp.symbols("q1 q2 r e")
    V_functional = (charge1 * charge2) / (position * electric_permetivity * 4 * sp.pi)

    def __init__(self, q1=1, q2=1, epsilon=1):
        """
        __init__
            This is the Constructor of the Coulomb potential
        Parameters
        ----------
        q1: int, optional
            Charge of atom 1, defaults to 1
        q2: int, optional
            Charge of atom 2, defaults to 1
        epsilon: int, optional
            Electric Permetivitty, defaults to 1
        """

        self.constants = {self.charge1: q1, self.charge2: q2, self.electric_permetivity: epsilon}
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


class lennardJonesPotential(_potential1DCls):
    """
     Lennard Jones potential representing the pairwise van-der-Waals interaction of two particles
     """
    name: str = "Lennard Jones Potential"
    sigma, epsilon, x_shift, y_shift, position = sp.symbols("s e r_0 V_off r")
    V_functional = 4 * epsilon * ((sigma / (position - x_shift)) ** 12 - (sigma / (position - x_shift)) ** 6) + y_shift

    def __init__(self, sigma: float = 1.5, epsilon: float = 2, x_shift: float = 0, y_shift=0):
        """
        __init__
            This is the Constructor of the Lennard-Jones Potential

        Parameters
        ----------
        sigma: float, optional
            x - Position of the minimum, defaults to 1.5
        epsilon: float, optional
            y - position of minimum, defaults to 2
        x_shift: float, optional
            shift of potential on x Axis, defaults to 0
        y_shift: int, optional
            shift of potential on y Axis, defaults to 0
        """

        self.constants = {self.sigma: sigma, self.epsilon: epsilon, self.x_shift: x_shift, self.y_shift: y_shift}

        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


class lennardJonesForceFieldPotential(_potential1DCls):
    """
            This is a forcefield like implementation of  a lennard Jones Potential
    """
    name: str = "Lennard Jones Potential"
    c6, c12, x_shift, y_shift, position = sp.symbols("c6 c12 r_0 V_off r")
    V_functional = (c12 / (position - x_shift) ** 12) - (c6 / (position - x_shift ** 6)) + y_shift

    def __init__(self, c6: float = 0.2, c12: float = 0.0001, x_shift: float = 0, y_shift: float = 0):
        """
        __init__
            This is the Constructor of the Lennard-Jones Field Potential
        Parameters
        ----------
        c6: float, optional
            prefactor of the interaction term that scales with **6, defaults to 0.2
        c12: float, optional
            prefactor of the interaction term that scales with **12, defaults to 0.0001
        x_shift: float, optional
            shift of potential on x Axis, defaults to 0
        y_shift: float, optional
            shift of potential on y Axis, defaults to 0
        """
        self.constants = {self.c6: c6, self.c12: c12, self.x_shift: x_shift, self.y_shift: y_shift}
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


class doubleWellPotential(_potential1DCls):
    """
            This is an implementation of a double Well potential
    """
    name: str = "Double Well"
    a, b, Vmax, position = sp.symbols("a b V_max r")
    V_functional = (Vmax / (b ** 4)) * ((position - a / 2) ** 2 - b ** 2) ** 2

    def __init__(self, Vmax=5, a=-1, b=1):
        """
        __init__
            This is the Constructor of the double well Potential

        Parameters
        ----------
        Vmax: int, optional
            Maximal barrier between minima, defaults to 5
        a: int, optional
            defines x position of the minimum of the first well, defaults to -1
        b: int, optional
            defines x position of the minimum of the second well, defaults to 1
        """

        self.constants = {self.Vmax: Vmax, self.a: a, self.b: b}
        super().__init__()
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)


class fourWellPotential(_potential1DCls):
    '''
        Unperturbed four well potential
    '''
    name: str = "Four Well Potential"

    a, ah, b, bh, c, ch, d, dh, Vmax, position = sp.symbols("a ah b bh c ch d dh V_max r")

    V_functional = -Vmax * sp.log(sp.exp(-(position - a) ** 2 - ah) + sp.exp(-(position - b) ** 2 - bh) + sp.exp(
        -(position - c) ** 2 - ch) + sp.exp(-(position - d) ** 2 - dh))

    def __init__(self, Vmax=4, a=1.5, b=4.0, c=7.0, d=9.0, ah=2., bh=0., ch=0.5, dh=1.):
        '''
        __init__
            This is the Constructor of the four well Potential

        Parameters
        ----------
        Vmax: float, optional
            scaling of the whole potential
        a: float, optional
            x position of the minimum of the first well
        b: float, optional
            x position of the minimum of the second well
        c: float, optional
            x position of the minimum of the third well
        d: float, optional
            x position of the minimum of the fourth well
        ah: str, optional
            ah*Vmax = y position of the first well
        bh: str, optional
            bh*Vmax = y position of the second well
        ch: str, optional
            ch*Vmax = y position of the third well
        dh: str, optional
            dh*Vmax = y position of the fourth well
        '''

        self.constants = {self.Vmax: Vmax, self.a: a, self.b: b, self.c: c, self.d: d, self.ah: ah, self.bh: bh,
                          self.ch: ch, self.dh: dh}
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


class gaussPotential(_potential1DCls):
    '''
        Gaussian like potential, usually used for metadynamics
    '''
    name: str = "Gaussian Potential"

    mu, sigma, A, position = sp.symbols("mu sigma A r")

    V_functional = A * sp.exp(-(position - mu) ** 2 / (2 * sigma ** 2))

    def __init__(self, A=1., mu=0., sigma=1.):
        '''
        __init__
            This is the Constructor of a 1D Gauss Potential

        Parameters
        ----------
        A: float, optional
            scaling of the gauss function, defaults to 1.
        mu: float, optional
            mean of the gauss function, defautls to 0.
        sigma: float, optional
            standard deviation of the gauss function, defaults to 1.

                TODO: make numerical stable
        '''

        self.constants = {self.A: A, self.mu: mu, self.sigma: sigma}
        super().__init__()

    #overwrite update_functions to remove the self.V = self.V_functional.subs(self.constants).expand() bug
    def _update_functions(self):
        """
        This function is needed to simplyfiy the symbolic equation on the fly and to calculate the position derivateive.
        """

        self.V = self.V_functional.subs(self.constants) #expand does not work with gaussians because of exp

        self.dVdpos_functional = sp.diff(self.V_functional, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")


"""
    COMBINED POTENTIALS
"""


class torsionPotential(_potential1DCls):
    """
    Torsion potential that represents the energy potential of a torsion angle
    """
    name: str = "Torsion Potential"

    phase: float = 1.0
    position = sp.symbols("r")
    wavePotentials = sp.Array([1])
    i, N = sp.symbols("i N")  # sum symbols
    V_functional = sp.Sum(wavePotentials[i, 0], (i, 0, N))

    def __init__(self, wavePotentials=[wavePotential(), wavePotential(multiplicity=3)], radians=False):
        """
        __init__
            This is the Constructor of a Torsion Potential

        Parameters
        ----------
        wavePotentials: list of two potentialTypes, optionel
            Torsion potential use the wave potential class to generate its potential, default to
            [wavePotential(), wavePotential(multiplicity=3)]
        radians: bool, optional
            set potential to radians or degrees, defaults to False
        """
        '''
        initializes torsions Potential
        '''
        wavePotentials = np.array(wavePotentials, ndmin=1)
        self.constants = {**{"wave_" + str(key): wave.V for key, wave in enumerate(wavePotentials)},
                          **{self.N: len(wavePotentials) - 1}}
        self.wavePotentials = sp.Matrix([sp.symbols("wave_" + str(i)) for i in range(len(wavePotentials))])
        self.V_functional = sp.Sum(self.wavePotentials[self.i, 0], (self.i, 0, self.N))

        super().__init__()
        self.set_radians(radians=radians)

    # OVERRIDE
    def _update_functions(self):
        """
        _update_functions
            calculates the current energy and derivative of the energy
        """
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos

    def set_degrees(self, degrees: bool = True):
        """
        Sets output to either degrees or radians

        Parameters
        ----------
        degrees: bool, optional,
            if True, output will be given in degrees, otherwise in radians, default: True
        """
        self.radians = not degrees
        if (degrees):
            self._calculate_energies = lambda positions: self.tmp_Vfunc(np.deg2rad(positions))
            self._calculate_dVdpos = lambda positions: self.tmp_dVdpfunc(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        """
        Sets output to either degrees or radians

        Parameters
        ----------
        radians: bool, optional,
            if True, output will be given in radians, otherwise in degree, default: True
        """
        self.radians = radians
        if (radians):
            self._calculate_energies = self.tmp_Vfunc
            self._calculate_dVdpos = self.tmp_dVdpfunc
        else:
            self.set_degrees(degrees=not radians)


class forceField:
    """
    Force field potential energy that combines Coulomb, Lennard Jones and Torsion potentials
    """

    def __init__(self):
        raise NotImplementedError("Not implemented yet, but this class shall be used to link N potential terms! ")


"""
    Multi State Potentials - PERTURBED POTENTIALS
"""


class linearCoupledPotentials(_potential1DClsPerturbed):
    """
    Linear Coupled Potential combines two potential as linear combinations,
    $ V_{\lambda} = \lambda * V_a + (1-\lambda)*V_b $

    This variant of coupling states is used for example in FEP, TI or BAR approaches.

    """
    name: str = "Linear Coupled System"
    lam, position = sp.symbols('λ r')
    Va, Vb = (sp.symbols("V_a"), sp.symbols("V_b"))
    coupling = (1 - lam) * Va + lam * Vb

    def __init__(self, Va: _potential1DCls = harmonicOscillatorPotential(k=1.0, x_shift=0.0),
                 Vb: _potential1DCls = harmonicOscillatorPotential(k=11.0, x_shift=0.0),
                 lam: float = 0.5):
        """
            __init__
                This constructor builds a linear combination of Va and Vb potentials, with lam as a cofactor.
                Linear Coupled Potentials, like in FEP or TI simulations.]

        Parameters
        ----------
        Va: _potential1DCls, optional
            Potential A that is mixed to the new potential.
        Vb:  _potential1DCls, optional
            Potential B that is mixed to the new potential.
        lam: float
            lam is representing the lambda variable
        """

        self.statePotentials = {self.Va: Va, self.Vb: Vb}
        self.constants = {self.Va: Va.V, self.Vb: Vb.V, self.lam: lam}
        super().__init__()


class exponentialCoupledPotentials(_potential1DCls):
    """
    This implementation of exponential Coupling is the symbolic variant of the more robust eds potential implementation.
    Here N-states are coupled by the log-sum-exp resulting in a new reference state $V_R$,

    $V_R = -1/{\beta} * \ln(\sum_i^Ne^(-\beta*s*(V_i-E^R_i)))$

    This potential coupling is for example used in EDS.

    """
    name: str = "exponential Coupled System"
    position, s, temp, eoffA, eoffB = sp.symbols('r s T eoffI eoffJ')
    Va, Vb = (sp.symbols("V_a"), sp.symbols("V_b"))
    beta = const.gas_constant / 1000.0 * temp
    coupling = -1 / (beta * s) * sp.log(sp.exp(-beta * s * Vb - eoffA) + sp.exp(-beta * s * Va - eoffB))

    def __init__(self, Va: _potential1DCls = harmonicOscillatorPotential(k=1.0, x_shift=0.0),
                 Vb: _potential1DCls = harmonicOscillatorPotential(k=11.0, x_shift=0.0),
                 eoffA: float = 0, eoffB: float = 0, s: float = 1.0, temp: float = 298):
        """
            __init__
                This constructor is building a exponential coupled Potential out of two given end-states.

        Parameters
        ----------
        Va: _potential1DCls, optional
            potential function of state A (default: harmonic oscillator)
        Vb: _potential1DCls, optional
            potential function of state B (default: harmonic oscillator)
        eoffA: float, optional
            Energy offset of state A in the reference potential (default: 0)
        eoffB: float, optional
            Energy offset of state B in the reference potential (default: 0)
        s: float, optional
            smoothing factor of the reference potential (default: 1.0)
        temp: float, optional
            Temperature of the reference state. (default: 298)

        """

        self.statePotentials = {self.Va: Va, self.Vb: Vb}
        self.constants = {self.Va: Va.V, self.Vb: Vb.V, self.eoffA: eoffA, self.eoffB: eoffB, self.s: s,
                          self.temp: temp}
        self.V_functional = self.coupling.expand()

        super().__init__(nStates=2)

    def set_s(self, s: float):
        """
            set_s
                sets a new s-value. (please only use this function to change s)

        Parameters
        ----------
        s: float
            the new sval.

        """
        self.constants.update({self.s: s})
        self._update_functions()

    def set_Eoff(self, eoffA: float = 0, eoffB: float = 0):
        """
            set_Eoff
                set the energy offsets for the states in the reference state.

        Parameters
        ----------
        eoffA: float, optional
            set a new offset for state A (default: None)
        eoffB: float, optional
            set a new E offset for state B in the reference state (default: None)

        """
        if (eoffA is None):
            self.constants.update({self.eoffA: eoffA})
        if (eoffB is None):
            self.constants.update({self.eoffB: eoffB})
        self._update_functions()


class envelopedPotential(_potential1DCls):
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
    V_functional = -1 / (beta * sis[0, 0]) * sp.log(
        sp.Sum(sp.exp(-beta * sis[i, 0] * (Vis[i, 0] - Eoffis[i, 0])), (i, 0, nStates)))

    def __init__(self, V_is: t.List[_potential1DCls] = (
            harmonicOscillatorPotential(), harmonicOscillatorPotential(x_shift=3)),
                 s: float = 1.0, eoff: t.List[float] = None, T: float = 1, kb: float = 1):
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
        self.constants.update({self.T: T, self.kb: kb})
        nStates = len(V_is)
        self._Eoff_i = [0 for x in range(nStates)]
        self._s = [0 for x in range(nStates)]
        self._V_is = [0 for x in range(nStates)]

        # for calculate implementations
        self.V_is = V_is
        self.s_i = s
        self.Eoff_i = eoff

        super().__init__(nStates=len(V_is))

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

        self.V_functional = -1 / (self.beta * self.sis[0, 0]) * sp.log(
            sp.Sum(sp.exp(-self.beta * self.states[self.i, 0]), (self.i, 0, self.nStates - 1)))
        self._update_functions()

        # also make sure that states are up to work:
        [V._update_functions() for V in self.V_is]

        if (all([self.s_i[0] == s for s in self.s_i[1:]])):
            self.ene = self._calculate_energies_singlePos_overwrite_oneS
        else:
            self.ene = self._calculate_energies_singlePos_overwrite_multiS
        self.force = self._calculate_dvdpos_singlePos_overwrite

    @property
    def V_is(self) -> t.List[_potential1DCls]:
        """
        V_is are the state potential classes enveloped by the reference state.

        Returns
        -------
        V_is: t.List[_potential1DCls]
        """
        return self._V_is

    @V_is.setter
    def V_is(self, V_is: t.List[_potential1DCls]):
        if (isinstance(V_is, Iterable) and all([isinstance(Vi, _potential1DCls) for Vi in V_is])):
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
        if (isinstance(Eoff, type(None))):
            self._Eoff_i = [0.0 for state in range(self.constants[self.nStates])]
            Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**Eoffis})
        elif (len(Eoff) == self.constants[self.nStates]):
            self._Eoff_i = Eoff
            Eoffis = {"Eoff_" + str(i): self.Eoff_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**Eoffis})
        else:
            raise IOError(
                "Energy offset Vector and state potentials don't have the same length!\n states in Eoff " + str(
                    len(Eoff)) + "\t states in Vi" + str(len(self.V_is)))

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
        if (isinstance(s, Number)):
            self._s = [s for x in range(self.constants[self.nStates])]
            sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**sis})
        elif (len(s) == self.constants[self.nStates]):
            raise NotImplementedError("Currently Only one s runs supported!")
            # self._s = s
            # self.constants.update({self.sis: self._s})
            # sis = {"s_" + str(i): self.s_i[i] for i in range(self.constants[self.nStates])}
            # self.constants.update({**sis})
        else:
            raise IOError("s Vector/Number and state potentials don't have the same length!\n states in s " + str(
                len(s)) + "\t states in Vi" + str(len(self.V_is)))

    def _calculate_energies_singlePos_overwrite_multiS(self, position) -> np.array:
        sum_prefactors, _ = self._logsumexp_calc_gromos(position)
        beta = self.constants[self.T] * self.constants[self.kb]  # kT - *self.constants[self.T]
        Vr = (-1 / (beta)) * sum_prefactors
        return np.squeeze(Vr)

    def _calculate_energies_singlePos_overwrite_oneS(self, position) -> np.array:
        sum_prefactors, _ = self._logsumexp_calc(position)
        beta = self.constants[self.T] * self.constants[self.kb]
        Vr = (-1 / (beta * self.s_i[0])) * sum_prefactors
        return np.squeeze(Vr)

    def _calculate_dvdpos_singlePos_overwrite(self, position: (t.Iterable[float])) -> np.array:
        """
            Todo: improve numerical stability.
        Parameters
        ----------
        position

        Returns
        -------

        """
        position = np.array(position, ndmin=2)
        # print("Pos: ", position)

        V_R_part, V_Is_ene = self._logsumexp_calc_gromos(position)
        V_R_part = np.array(V_R_part, ndmin=2).T
        # print("V_R_part: ", V_R_part.shape, V_R_part)
        # print("V_I_ene: ",V_Is_ene.shape, V_Is_ene)
        V_Is_dhdpos = np.array([-statePot.force(position) for statePot in self.V_is], ndmin=1).T
        # print("V_I_force: ",V_Is_dhdpos.shape, V_Is_dhdpos)

        adapt = np.concatenate([V_R_part for s in range(self.constants[self.nStates])], axis=1)
        # print("ADAPT: ",adapt.shape, adapt)
        scaling = np.exp(V_Is_ene - adapt)
        # print("scaling: ", scaling.shape, scaling)
        dVdpos_state = np.multiply(scaling,
                                   V_Is_dhdpos)  # np.array([(ene/V_R_part) * force for ene, force in zip(V_Is_ene, V_Is_dhdpos)])
        # print("state_contributions: ",dVdpos_state.shape, dVdpos_state)
        dVdpos = np.sum(dVdpos_state, axis=1)
        # print("forces: ",dVdpos.shape, dVdpos)

        return np.squeeze(dVdpos)

    def _logsumexp_calc(self, position):
        prefactors = []
        beta = self.constants[self.T] * self.constants[self.kb]
        for state in range(self.constants[self.nStates]):
            prefactor = np.array(-beta * self.s_i[state] * (self.V_is[state].ene(position) - self.Eoff_i[state]),
                                 ndmin=1).T
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
            sum_prefactors = np.max([sum_prefactors, partN], axis=1) + np.log(1 + np.exp(
                np.min([sum_prefactors, partN], axis=1) - np.max([sum_prefactors, partN], axis=1)))
            # print("prefactors: ", sum_prefactors)
        return sum_prefactors, np.array(prefactors, ndmin=2).T


class hybridCoupledPotentials(_potential1DClsPerturbed):
    """
    This implementation of exponential Coupling combined with linear compling is called $\lambda$-EDS this function is the purely symbolic class and therefore note very numerical stable (see lambda EDS).

    Here two-states are coupled by the log-sum-exp and weighted by lambda resulting in a new reference state $V_R$,

    $V_R = -\frac{1}{\beta*s} * \ln(\lambda * e^(-\beta*s*(V_A-E^R_A)) + (1-\lambda)*e^(-\beta*s*(V_B-E^R_B)))$

    This potential coupling is for example used in $\lambda$-EDS.
    """

    name: str = "hybrid Coupled Potential"
    lam, position, s, T = sp.symbols(u'λ r s T')
    Va, Vb = (sp.symbols("V_a"), sp.symbols("V_b"))
    beta = 1  # const.gas_constant / 1000.0 * temp
    coupling = -1 / (beta * s) * sp.log(lam * sp.exp(-beta * s * Vb) + (1 - lam) * sp.exp(-beta * s * Va))

    def __init__(self, Va: _potential1DCls = harmonicOscillatorPotential(k=1.0, x_shift=0.0),
                 Vb: _potential1DCls = harmonicOscillatorPotential(k=11.0, x_shift=0.0),
                 lam: float = 0.5, s: float = 1.0, temp: float = 298):
        """
            __init__
                This function constructs a $\lambda$-enveloped potential, enveloping all given states and weighting them by $\lambda$.

        Parameters
        ----------
        V_is: List[_potential1DCls], optional
            The states(potential classes) to be enveloped (default: [harmonicOscillatorPotential(), harmonicOscillatorPotential(x_shift=3)])

        s: float, optional
            the smoothing parameter, lowering the barriers between the states
        Eoff_i: List[float], optional
            the energy offsets of the individual states in the reference potential. These can be used to allow a more uniform sampling. (default: seta ll to 0)
        T: float, optional
            the temperature of the reference state (default: 1 = T)
        kb: float, optional
            the boltzman constant (default: 1 = kb)

        """

        self.statePotentials = {self.Va: Va, self.Vb: Vb}
        self.constants = {self.Va: Va.V, self.Vb: Vb.V, self.lam: lam, self.s: s, self.T: temp}

        super().__init__()

    def set_s(self, s: float):
        self.constants.update({self.s: s})
        self._update_functions()

    def set_Eoff(self, Eoff: float):
        self.constants.update({self.Eoff: Eoff})
        self._update_functions()


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
    V_functional = -1 / (beta * sis[0, 0]) * sp.log(
        sp.Sum(lamis[i, 0] * sp.exp(-beta * sis[i, 0] * (Vis[i, 0] - Eoffis[i, 0])), (i, 0, nStates)))

    def __init__(self, V_is: t.List[_potential1DCls] = (
            harmonicOscillatorPotential(), harmonicOscillatorPotential(x_shift=3)), lam: Number = 0.5,
                 s: float = 1.0, eoff: t.List[float] = None, T: float = 1, kb: float = 1):

        nStates = len(V_is)
        self.constants.update({self.nStates: nStates})
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
        self.constants.update(
            {**{state: value.V for state, value in self.statePotentials.items()}, **Eoffis, **sis, **lamis})
        inner_log = sp.Sum(sp.Matrix(list(lamis.keys()))[self.i, 0] * sp.exp(-self.beta * self.states[self.i, 0]),
                           (self.i, 0, self.nStates - 1))
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
        if (isinstance(lam, Number) and self.constants[self.nStates] == 2):
            self._lam_i = np.array([lam] + [1 - lam for x in range(1, self.constants[self.nStates])], ndmin=1)
            lamis = {"lam_" + str(i): self.lam_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**lamis})
        elif (isinstance(lam, Number)):  # a bit klunky here! TODO: think about a nice solution
            self._lam_i = np.array([1 / self.constants[self.nStates] for x in range(self.constants[self.nStates])],
                                   ndmin=1)
            lamis = {"lam_" + str(i): self.lam_i[i] for i in range(self.constants[self.nStates])}
            self.constants.update({**lamis})
        elif (len(lam) == self.constants[self.nStates]):
            raise NotImplementedError("Currently Only one lam runs supported!")
            # self._lam_i = np.array(lam, ndmin=1)
            # self.constants.update({self.lamis: self._lam_i})
        else:
            raise IOError("s Vector/Number and state potentials don't have the same length!\n states in s " + str(
                lam) + "\t states in Vi" + str(len(self.V_is)))

    def _calculate_energies_singlePos_overwrite(self, position) -> np.array:
        # print("Positions: ",position)
        # print("s_i: ",self.s_i)
        sum_prefactors, _ = self._logsumexp_calc(position)
        beta = self.constants[self.T] * self.constants[self.kb]  # kT - *self.constants[self.T]
        Vr = (-1 / (beta * self.s_i[0])) * sum_prefactors
        # print("finalVR", Vr)
        return np.squeeze(Vr)

    def _calculate_dvdpos_singlePos_overwrite(self, position: (t.Iterable[float])) -> np.array:
        position = np.array(position, ndmin=2)
        # print("Pos: ", position)
        V_R_part, V_Is_ene = self._logsumexp_calc(position)
        # print("V_I_ene: ",V_Is_ene.shape, V_Is_ene)
        V_R_part = np.array(V_R_part, ndmin=2).T
        # print("V_R_part: ", V_R_part.shape, V_R_part)
        V_Is_dhdpos = np.array([-statePot.force(position) for statePot in self.V_is], ndmin=1).T
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
            prefactor = np.array(-beta * self.s_i[state] * (self.V_is[state].ene(position) - self.Eoff_i[state]),
                                 ndmin=1).T
            prefactors.append(prefactor)
        prefactors = np.array(prefactors, ndmin=2).T

        from scipy.special import logsumexp
        sum_prefactors = logsumexp(prefactors, axis=1, b=self.lam)

        return np.squeeze(sum_prefactors), np.array(prefactors, ndmin=2).T


"""
special potentials
"""


class dummyPotential(_potential1DCls):
    """
    This Dummy potential returns a simple constant value for each position
    """
    name: str = "Dummy Potential"
    position, y_shift = sp.symbols("r Voffset")

    def __init__(self, y_shift: float = 0):
        """
        This Class is representing the a dummy potential.
        It returns a constant value equalling the y_shift parameter.


        Parameters
        ----------
        y_shift: float, optional
            This will be the constant return value, defaults to 0

        """

        self.V_functional = sp.Lambda(self.position, self.y_shift)

        self.constants = {self.y_shift: y_shift}
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)
        super().__init__()

        self._calculate_energies = lambda positions: np.squeeze(np.full(len(positions), y_shift))
        self.dVdpos = self._calculate_dVdpos = lambda positions: np.squeeze(np.zeros(len(positions)))


class flatwellPotential(_potential1DCls):
    """
    A flatwell potential returns a simple constant value (y_max) for each position except positions in the range x_min-x_max.
    In the defined phase space range the potential always returns a second defined value (y_min)
    """
    name: str = "Flat Well"

    x_min: float = None
    x_max: float = None
    y_max: float = None
    y_min: float = None

    def __init__(self, x_range: list = (0, 1), y_max: float = 4, y_min: float = 0):
        """
        __init__ This potential is a flatwell potential.
        The flatwell potential is a function similar to an if case.
        If a position is inside a the x_range, it returns the y_min val.
        If a position is outside, the y_max val will be returned.

        Parameters
        ----------
        x_range:  (list, range), optional
        range inside this the y_min val will be returned, defaults to (0, 1)
        y_max:  float, optional
        outside of the range this value will be returned, defaults to 1000
        y_min: float, optional
        inside the range this value will be returned, defaults to 0
        """

        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

        self.constants.update({self.nStates: 1, self.nDimensions: 1})
        self._update_functions = None

    def _calculate_energies(self, position: Union[Number, np.array]) -> Union[Number, np.array]:
        return np.squeeze([self.y_min if (pos >= self.x_min and pos <= self.x_max) else self.y_max for pos in
                           np.array(np.squeeze(position), ndmin=1)])

    def _calculate_dVdpos(self, positions: Union[Number, np.array]) -> Union[Number, np.array]:
        return np.squeeze([np.inf if (pos == self.x_min or pos == self.x_max) else 0 for pos in
                           np.array(np.squeeze(positions), ndmin=1)])

    def __setstate__(self, state):
        """
        Setting up after pickling. Catch special features fo function
        """
        self.__dict__ = state


"""
Biased potentials
"""
"""
    TIME INDEPENDENT BIASES 
"""


class addedPotentials(_potential1DCls):
    '''
    Adds two different potentials on top of each other. Can be used to generate
    harmonic potential umbrella sampling or scaled potentials
    '''
    name: str = "Added Potential Enhanced Sampling System"
    position = sp.symbols("r")
    bias_potential = True

    def __init__(self, origPotential=harmonicOscillatorPotential(), addPotential=gaussPotential()):
        '''
        __init__
              This is the Constructor of the addedPotential class.
        Parameters
        ----------
        origPotential: potential type
            The unbiased potential
        addPotential: potential type
            The potential added on top of the unbiased potential to
            bias the system
        '''

        self.origPotential = origPotential
        self.addPotential = addPotential

        self.constants = {**origPotential.constants, **addPotential.constants}

        self.V_functional = origPotential.V + self.addPotential.V

        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


"""
    TIME DEPENDENT BIASES 
"""


class metadynamicsPotential(_potential1DCls):
    '''
    The metadynamics bias potential adds 1D Gaussian potentials on top of
    the original 1D potential. The added gaussian potential is centered on the current position.
    Thereby the valleys of the potential "flooded" and barrier crossing is easier.

    This implementation uses a grid to store the biasing. This is much faster than calculating
    an ever increasing potential with sympy
    '''
    name: str = "Metadynamics Enhanced Sampling System using grid bias"
    position = sp.symbols("r")
    bias_potential = True

    def __init__(self, origPotential=harmonicOscillatorPotential(), amplitude=0.1, sigma=1, n_trigger=100,
                 bias_grid_min=0, bias_grid_max=10,
                 numbins=100):

        '''
        This is the Constructor of the metadynamicsPotential class.
        Parameters
        ----------
        origPotential: potential type
            The unbiased potential
        amplitude : float
            scaling of the gaussian potential added in the metadynamcis step
        sigma: float
            standard deviation of the gaussian potential added in the metadynamcis step
        n_trigger : int
            Metadynamics potential will be added after every n_trigger'th steps
        bias_grid_min: float
            min value of the bias grid
        bias_grid_max: float
            max value of the bias grid
        numbins: float
            size of the grid bias and forces are saved in
        '''

        self.origPotential = origPotential
        self.n_trigger = n_trigger
        self.amplitude = amplitude
        self.sigma = sigma
        self.biasPotentialType = gaussPotential

        # grid where the bias is stored
        # currently only for 1D
        self.bias_grid_energy = np.zeros(numbins)  # energy grid
        self.bias_grid_force = np.zeros(numbins)  # force grid
        # get center value for each bin
        bin_half = (bias_grid_max - bias_grid_min) / (2 * numbins)  # half bin width
        self.bin_centers = np.linspace(bias_grid_min + bin_half, bias_grid_max - bin_half, numbins)
        # current_n counts when next metadynamic step should be applied
        self.current_n = 1
        # count how often the potential was updated
        self.finished_steps = 0

        self.constants = {**origPotential.constants}

        self.V_functional = origPotential.V
        self.V_orig_part = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V_orig_part, self.position)
        self.V = self.V_orig_part

        super().__init__()

    """
    BIAS
    """

    # Beautiful integration to system as Condition.
    def apply(self):
        self.check_for_metastep(self.system._currentPosition)

    def apply_coupled(self):
        self.check_for_metastep(self.system._currentPosition)

    def couple_system(self, system: systemCls):
        self.system = system

    def check_for_metastep(self, curr_position):
        '''
        Checks if the bias potential should be added at the current step
        Parameters
        ----------
        curr_position: tuple
            current x,y position

        Returns
        -------

        '''
        if (self.system.step % self.n_trigger == 0):
            self._update_potential(curr_position)
            self.finished_steps += 1

        """
        TODO: Remove
        if self.current_n%self.n_trigger == 0:
            self._update_potential(curr_position)
            self.current_n += 1
        else:
            self.current_n += 1
        """

    def _update_potential(self, curr_position):
        '''
        Is triggered by check_for_metastep(). Adds a gaussian centered on the
        current position to the potential

        Parameters
        ----------
        curr_position: float
            current x position

        Returns
        -------


        '''
        # do gaussian metadynamics
        # print("A ", self.amplitude, "mu ", curr_position, "sigma ", self.sigma)
        biasPotential = self.biasPotentialType(A=self.amplitude, mu=curr_position, sigma=self.sigma)
        try:
            new_bias_bin_energy = biasPotential.ene(self.bin_centers)
            new_bias_bin_force = biasPotential.force(self.bin_centers)
        except OverflowError:
            print("Gaussian Overflows!")

        # update bias grid
        self.bias_grid_energy = self.bias_grid_energy + new_bias_bin_energy
        self.bias_grid_force = self.bias_grid_force + new_bias_bin_force

    # overwrite the energy and force
    def ene(self, positions):
        '''
        calculates energy of particle also takes bias into account
        Parameters
        ----------
        positions: tuple
            position on 1D potential energy surface

        Returns
        -------
        current energy
        '''

        if isinstance(positions, float) or isinstance(positions, int):
            current_bin = self._find_nearest(self.bin_centers, positions)
            return np.squeeze(self._calculate_energies(np.squeeze(positions)) + self.bias_grid_energy[current_bin])
        else:
            bias_list = []
            for entry in positions:
                current_bin = self._find_nearest(self.bin_centers, entry)
                bias_list.append(self.bias_grid_energy[current_bin])
            return np.squeeze(self._calculate_energies(np.squeeze(positions)) + np.array(bias_list))

    def force(self, positions):
        '''
        calculates derivative with respect to position also takes bias into account

        Parameters
        ----------
        positions: tuple
            position on 1D potential energy surface

        Returns
        current derivative dh/dpos
        -------
        '''

        current_bin = np.apply_over_axes(self._find_nearest, a=np.array(positions, ndmin=1),
                                         axes=0)  # self._find_nearest(self.bin_centers, positions)
        force = np.squeeze(self._calculate_dVdpos(np.squeeze(positions)) + self.bias_grid_force[current_bin])
        return force

    def _find_nearest(self, array, value):
        '''
        Function that finds position of the closest entry to a given value in an array

        Parameters
        ----------
        array: np.array
            1D array containing the midpoints of the metadynamics grid
        value: int or float
            search value
        Returns

        Index of the entry closest to the given value
        -------

        '''
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
            return idx - 1
        else:
            return idx


#### OLD FUNCTIONS ###

class timedependendBias(_potential1DCls):
    '''
    The timedependend bias potential adds a user defined potential on top of
    the original potential.

    This implementation uses sympy instead of a grid and is therefore super slow
    '''
    name: str = "Metadynamics Enhanced Sampling System"
    position = sp.symbols("r")

    def __init__(self, origPotential, addPotential, n_trigger):

        '''
        __init__
              This is the Constructor of the addedPotential class.
        Parameters
        ----------
        origPotential: potential type
            The unbiased potential
        addPotential: potential type
            The potential added on top of the unbiased potential to
            bias the system, usually of gaussian type
        n_trigger : int
            Added potential will be added after every n_trigger'th steps
        '''
        self.origPotential = origPotential
        self.n_trigger = n_trigger
        self.addPotential = addPotential
        # current_n counts when next potential adding step should be applied
        self.current_n = 1

        self.constants = {**origPotential.constants, **addPotential.constants}

        self.V_orig = origPotential.V
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

    def check_for_metastep(self, curr_position):
        '''
        Checks if the bias potential should be added at the current step
        Parameters
        ----------
        curr_position: flaot
            current x position
        Returns
        -------
        '''
        if self.current_n % self.n_trigger == 0:
            self._update_potential()
            self.current_n = 1
        else:
            self.current_n += 1

    def _update_potential(self):
        '''
        Is triggered by check_for_metastep(). Adds the pre-defined potential on the
        current position to the potential

        Parameters
        ----------
        Returns
        -------
        '''
        # add potential to the system
        self.V_functional = self.V + self.addPotential.V
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)


class metadynamicsPotentialSympy(_potential1DCls):
    '''
    The metadynamics bias potential adds Gaussian potentials on top of
    the original potential. The added gaussian potential is centered on the current position.
    Thereby the valleys of the potential "flooded" and barrier crossing is easier

    This implementation uses sympy instead of a grid and is therefore super slow
    '''

    name: str = "Metadynamics Enhanced Sampling System using sympy"
    position = sp.symbols("r")

    def __init__(self, origPotential, amplitude=0.1, sigma=0.1, n_trigger=100):

        '''
        This is the Constructor of the metadynamicsPotential class.
        Parameters
        ----------
        origPotential: potential type
            The unbiased potential
        amplitude : float
            scaling of the gaussian potential added in the metadynamcis step
        sigma: float
            standard deviation of the gaussian potential added in the metadynamcis step
        n_trigger : int
            Metadynamics potential will be added after every n_trigger'th steps
        '''

        self.origPotential = origPotential
        self.biasPotential = gaussPotential
        self.n_trigger = n_trigger
        self.amplitude = amplitude
        self.sigma = sigma
        # current_n counts when next metadynamic step should be applied
        self.current_n = 1
        # count how often the potential was updated
        self.finished_steps = 0

        self.constants = {}
        self.constants.update(origPotential.constants)
        self.constants.update(self.biasPotential(A=self.amplitude, sigma=self.sigma).constants)

        self.V_orig = origPotential.V
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

    def check_for_metastep(self, curr_position):
        '''
        Checks if the bias potential should be added at the current step
        Parameters
        ----------
        curr_position: flaot
            current x position

        Returns
        -------
        '''
        if self.current_n % self.n_trigger == 0:
            self._update_potential(curr_position)
            self.finished_steps += 1
            self.current_n = 1
        else:
            self.current_n += 1

    def _update_potential(self, curr_position):
        '''
        Is triggered by check_for_metastep(). Adds a gaussian centered on the
        current position to the potential

        Parameters
        ----------
        curr_position: float
            current x position

        Returns
        -------
        '''
        # add potential to the system
        # do gaussian metadynamics
        self.V_functional = self.V + self.biasPotential(A=self.amplitude, mu=curr_position, sigma=self.sigma).V
        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)
