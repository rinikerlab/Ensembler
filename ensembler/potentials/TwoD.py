"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

from typing import List

import numpy as np
import sympy as sp

from ensembler.potentials._basicPotentials import _potential2DCls


class harmonicOscillatorPotential(_potential2DCls):
    """
        2D  harmonic oscillator potential
    """

    name: str = "harmonicOscilator"
    nDim: int = sp.symbols("nDim")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    r_shift: sp.Matrix = sp.Matrix([sp.symbols("r_shift")])
    Voff: sp.Matrix = sp.Matrix([sp.symbols("V_off")])
    k: sp.Matrix = sp.Matrix([sp.symbols("k")])

    V_dim = 0.5 * k * (position - r_shift) ** 2 + Voff

    i = sp.Symbol("i")
    V_functional = sp.Sum(V_dim[i, 0], (i, 0, nDim))

    def __init__(self, k: np.array = np.array([1.0, 1.0]), r_shift: np.array = np.array([0.0, 0.0]),
                 Voff: np.array = np.array([0.0, 0.0])):
        self.constants.update({self.nDim: 2})
        self.constants.update({"k_" + str(j): k[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"r_shift" + str(j): r_shift[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"V_off_" + str(j): Voff[j] for j in range(self.constants[self.nDim])})
        super().__init__()

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(self.constants[self.nDim])])
        self.r_shift = sp.Matrix([sp.symbols("r_shift" + str(i)) for i in range(self.constants[self.nDim])])
        self.V_off = sp.Matrix([sp.symbols("V_off_" + str(i)) for i in range(self.constants[self.nDim])])
        self.k = sp.Matrix([sp.symbols("k_" + str(i)) for i in range(self.constants[self.nDim])])
        # Function
        self.V_dim = 0.5 * sp.matrix_multiply_elementwise(self.k, (
            (self.position - self.r_shift).applyfunc(lambda x: x ** 2)))  # +self.Voff
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDim - 1))


class wavePotential(_potential2DCls):
    name: str = "Wave Potential"
    nDim: sp.Symbol = sp.symbols("nDim")

    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    multiplicity: sp.Matrix = sp.Matrix([sp.symbols("m")])
    phase_shift: sp.Matrix = sp.Matrix([sp.symbols("omega")])
    amplitude: sp.Matrix = sp.Matrix([sp.symbols("A")])
    yOffset: sp.Matrix = sp.Matrix([sp.symbols("y_off")])

    V_dim = sp.matrix_multiply_elementwise(amplitude,
                                           (sp.matrix_multiply_elementwise((position + phase_shift),
                                                                           multiplicity)).applyfunc(sp.cos)) + yOffset
    i = sp.Symbol("i")
    V_functional = sp.Sum(V_dim[i, 0], (i, 0, nDim))

    def __init__(self, amplitude=(1, 1), multiplicity=(1, 1), phase_shift=(0, 0), y_offset=(0, 0),
                 radians: bool = False):
        nDim = 2
        self.constants.update({"amp_" + str(j): amplitude[j] for j in range(nDim)})
        self.constants.update({"mult_" + str(j): multiplicity[j] for j in range(nDim)})
        self.constants.update({"yOff_" + str(j): y_offset[j] for j in range(nDim)})
        self.constants.update({"nDim": nDim})

        if (radians):
            self.constants.update({"phase_" + str(j): np.deg2rad(phase_shift[j]) for j in range(nDim)})
        else:
            self.constants.update({"phase_" + str(j): phase_shift[j] for j in range(nDim)})

        super().__init__()

        self.set_radians(radians=radians)

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDim)])
        self.multiplicity = sp.Matrix([sp.symbols("mult_" + str(i)) for i in range(nDim)])
        self.phase_shift = sp.Matrix([sp.symbols("phase_" + str(i)) for i in range(nDim)])
        self.amplitude = sp.Matrix([sp.symbols("amp_" + str(i)) for i in range(nDim)])
        self.yOffset = sp.Matrix([sp.symbols("yOff_" + str(i)) for i in range(nDim)])

        # Function
        self.V_dim = sp.matrix_multiply_elementwise(self.amplitude,
                                                    (sp.matrix_multiply_elementwise((self.position + self.phase_shift),
                                                                                    self.multiplicity)).applyfunc(
                                                        sp.cos)) + self.yOffset
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDim - 1))

    # OVERRIDE
    def _update_functions(self):
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos

    def set_degrees(self, degrees: bool = True):
        self.radians = not degrees
        if (degrees):
            self._calculate_energies = lambda positions, positions2: self.tmp_Vfunc(np.deg2rad(positions),
                                                                                    np.deg2rad(positions2))
            self._calculate_dVdpos = lambda positions, positions2: self.tmp_dVdpfunc(np.deg2rad(positions),
                                                                                     np.deg2rad(positions2))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            self._calculate_energies = self.tmp_Vfunc
            self._calculate_dVdpos = self.tmp_dVdpfunc
        else:
            self.set_degrees(degrees=not radians)


class torsionPotential(_potential2DCls):
    name: str = "Torsion Potential"

    position = sp.symbols("r")
    wave_potentials: sp.Matrix = sp.Matrix([sp.symbols("V_x")])

    nWavePotentials = sp.symbols("N")
    i = sp.symbols("i", cls=sp.Idx)

    V_functional = sp.Sum(wave_potentials[i, 0], (i, 0, nWavePotentials))

    def __init__(self, wave_potentials: List[wavePotential] = (wavePotential(), wavePotential(multiplicity=[3, 3])),
                 degrees: bool = True):
        '''
        initializes torsions Potential
        '''
        self.constants.update({self.nWavePotentials: len(wave_potentials)})
        self.constants.update({"V_" + str(i): wave_potentials[i].V for i in range(len(wave_potentials))})

        super().__init__()
        self.set_degrees(degrees=degrees)

    def _initialize_functions(self):
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(self.constants[self.nDim])])
        self.wave_potentials = sp.Matrix(
            [sp.symbols("V_" + str(i)) for i in range(self.constants[self.nWavePotentials])])
        # Function
        self.V_functional = sp.Sum(self.wave_potentials[self.i, 0], (self.i, 0, self.nWavePotentials - 1))

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
        msg += "\n"
        return msg

    # OVERRIDE
    def _update_functions(self):
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos

    def set_degrees(self, degrees: bool = True):
        self.radians = not degrees
        if (degrees):
            self._calculate_energies = lambda positions, positions2: self.tmp_Vfunc(np.deg2rad(positions),
                                                                                    np.deg2rad(positions2))
            self._calculate_dVdpos = lambda positions, positions2: self.tmp_dVdpfunc(np.deg2rad(positions),
                                                                                     np.deg2rad(positions2))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            self._calculate_energies = self.tmp_Vfunc
            self._calculate_dVdpos = self.tmp_dVdpfunc
        else:
            self.set_degrees(degrees=not radians)


class gaussPotential(_potential2DCls):
    '''
        Gaussian like potential, usually used for metadynamics
    '''
    name: str = "Gaussian Potential 2D"
    nDim: sp.Symbol = sp.symbols("nDim")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    mean: sp.Matrix = sp.Matrix([sp.symbols("mu")])
    sigma: sp.Matrix = sp.Matrix([sp.symbols("sigma")])
    amplitude = sp.symbols("A_gauss")

    # we assume that the two dimentions are uncorrelated
    #V_dim = amplitude * (sp.matrix_multiply_elementwise((position - mean) ** 2, (2 * sigma ** 2) ** (-1)).applyfunc(sp.exp))
    V_dim = amplitude * (sp.matrix_multiply_elementwise(-(position - mean).applyfunc(lambda x: x ** 2),
                                                        0.5 * (sigma).applyfunc(lambda x: x ** (-2))).applyfunc(sp.exp))

    i = sp.Symbol("i")
    V_functional = sp.product(V_dim[i, 0], (i, 0, nDim))

    # V_orig = V_dim[0, 0] * V_dim[1, 0]

    def __init__(self, amplitude=1., mu=(0., 0.), sigma=(1., 1.)):
        '''
        Parameters
        ----------
        A: float
            scaling of the gauss function
        mu: tupel
            mean of the gauss function
        sigma: tupel
            standard deviation of the gauss function
        '''
        nDim = 2
        self.constants.update({"A_gauss": amplitude})
        self.constants.update({"mu_" + str(j): mu[j] for j in range(nDim)})
        self.constants.update({"sigma_" + str(j): sigma[j] for j in range(nDim)})
        self.constants.update({"nDim": nDim})

        super().__init__()

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDim)])
        self.mean = sp.Matrix([sp.symbols("mu_" + str(i)) for i in range(nDim)])
        self.sigma = sp.Matrix([sp.symbols("sigma_" + str(i)) for i in range(nDim)])
        self.amplitude = sp.symbols("A_gauss")

        # Function
        self.V_dim = self.amplitude * (
            sp.matrix_multiply_elementwise(-(self.position - self.mean).applyfunc(lambda x: x ** 2),
                                           0.5 * (self.sigma).applyfunc(lambda x: x ** (-2))).applyfunc(sp.exp))

        # self.V_functional = sp.Product(self.V_dim[self.i, 0], (self.i, 0, self.nDim - 1))
        # Not too beautiful, but sp.Product raises errors
        self.V_functional = self.V_dim[0, 0] * self.V_dim[1, 0]


"""
Biased potentials
"""
from ensembler.potentials.biased_potentials.biasTwoD import *