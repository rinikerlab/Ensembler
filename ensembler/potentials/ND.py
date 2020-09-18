"""
Module: Potential
    This module shall be used to implement subclasses of Potentials that formulate a potential as an Function with N-Dimensions.
     This module contains all available potentials.
"""

import numpy as np
import sympy as sp

# Base Classes
from ensembler.potentials._basicPotentials import _potentialNDCls

class harmonicOscillatorPotential(_potentialNDCls):
    """
        ND  harmonic oscillator potential
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

    def __init__(self, k: np.array = np.array([1.0, 1.0, 1.0]), r_shift: np.array = np.array([0.0, 0.0, 0.0]),
                 Voff: np.array = np.array([0.0, 0.0, 0.0]), nDim: int = 3):
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
        super().__init__(nDim=nDim)
        self.constants.update({"k_" + str(j): k[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"r_shift" + str(j): r_shift[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"V_off_" + str(j): Voff[j] for j in range(self.constants[self.nDim])})

    def _initialize_functions(self):
        """
        Build up the nDim symbolic definitions
        """
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDim)])
        self.r_shift = sp.Matrix([sp.symbols("r_shift" + str(i)) for i in range(nDim)])
        self.V_off = sp.Matrix([sp.symbols("V_off_" + str(i)) for i in range(nDim)])
        self.k = sp.Matrix([sp.symbols("k_" + str(i)) for i in range(nDim)])
        # Function
        self.V_dim = 0.5 * sp.matrix_multiply_elementwise(self.k, (
            (self.position - self.r_shift).applyfunc(lambda x: x ** 2)))  # +self.Voff
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDim - 1))
