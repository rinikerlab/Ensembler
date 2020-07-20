"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""
import os

import numpy as np
import sympy as sp
import math
from numbers import Number
from typing import Iterable, List, Sized, Union

from ensembler.potentials._baseclasses import _potential2DCls, _potential2DClsSymPY
from ensembler.potentials.ND import envelopedPotential


class harmonicOscillator(_potential2DClsSymPY):
    '''
        .. autoclass:: harmonic oscillator potential
    '''
    name:str = "harmonicOscilator"
    x_shift = None
    fc = None
    nDim:int = sp.symbols("nDim")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    r_shift: sp.Matrix = sp.Matrix([sp.symbols("r_shift")])
    Voff: sp.Matrix = sp.Matrix([sp.symbols("V_off")])
    k: sp.Matrix = sp.Matrix([sp.symbols("k")])

    V_dim =  0.5*k*(position-r_shift)**2+Voff
    i = sp.Symbol("i")
    V_orig = sp.Sum(V_dim[i, 0], (i, 0, nDim))

    def __init__(self, k: np.array =  np.array([1.0, 1.0]), r_shift: np.array = np.array([0.0, 0.0]), Voff: np.array =  np.array([0.0, 0.0])):
        '''
        initializes harmonicOsc1D class
        :param fc: force constant
        :param x_shift: minimum position of harmonic oscillator
        '''
        #if(any(self.nDim != len(x) for x in [k, r_shift, Voff])):
        #    raise ValueError("All parameters need to be iterable and have len(x) == 2!")

        self.constants.update({self.nDim:2})
        self.constants.update({"k_"+str(j): k[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"r_shift"+str(j): r_shift[j] for j in range(self.constants[self.nDim])})
        self.constants.update({"V_off_"+str(j): Voff[j] for j in range(self.constants[self.nDim])})
        super().__init__()


    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(self.constants[self.nDim])])
        self.r_shift = sp.Matrix([sp.symbols("r_shift" + str(i)) for i in range(self.constants[self.nDim])])
        self.V_off = sp.Matrix([sp.symbols("V_off_" + str(i)) for i in range(self.constants[self.nDim])])
        self.k = sp.Matrix([sp.symbols("k_" + str(i)) for i in range(self.constants[self.nDim])])
        #Function
        self.V_dim =   0.5*sp.matrix_multiply_elementwise(self.k, ((self.position-self.r_shift).applyfunc(lambda x: x**2)))#+self.Voff
        self.V_orig = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDim-1))


class wavePotential(_potential2DClsSymPY):
    name:str = "Wave Potential"
    nDim:sp.Symbol = sp.symbols("nDim")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    multiplicity: sp.Matrix = sp.Matrix([sp.symbols("m")])
    phase_shift: sp.Matrix = sp.Matrix([sp.symbols("omega")])
    amplitude: sp.Matrix = sp.Matrix([sp.symbols("A")])
    yOffset: sp.Matrix = sp.Matrix([sp.symbols("y_off")])
    V_dim = sp.matrix_multiply_elementwise(amplitude,
                                        (sp.matrix_multiply_elementwise((position + phase_shift), multiplicity)).applyfunc(sp.cos)) + yOffset
    i = sp.Symbol("i")
    V_orig = sp.Sum(V_dim[i, 0], (i, 0, nDim))

    def __init__(self, amplitude=(1,1), multiplicity=(1,1), phase_shift=(0,0), y_offset=(0, 0), degree:bool=True):
        nDim = 2 
        self.constants.update({"amp_"+str(j): amplitude[j] for j in range(nDim)})
        self.constants.update({"mult_"+str(j): multiplicity[j] for j in range(nDim)})
        self.constants.update({"yOff_"+str(j): y_offset[j] for j in range(nDim)})
        self.constants.update({"nDim": nDim})

        if(degree):
            self.constants.update({"phase_"+str(j): np.deg2rad(phase_shift[j]) for j in range(nDim)})
        else:
            self.constants.update({"phase_"+str(j): phase_shift[j] for j in range(nDim)})

        
        super().__init__()

        if(degree):
            self.set_degree_mode()

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("pos_" + str(i)) for i in range(nDim)])
        self.multiplicity = sp.Matrix([sp.symbols("mult_" + str(i)) for i in range(nDim)])
        self.phase_shift = sp.Matrix([sp.symbols("phase_" + str(i)) for i in range(nDim)])
        self.amplitude = sp.Matrix([sp.symbols("amp_" + str(i)) for i in range(nDim)])
        self.yOffset = sp.Matrix([sp.symbols("yOff_" + str(i)) for i in range(nDim)])

        #Function
        self.V_dim = sp.matrix_multiply_elementwise(self.amplitude,
                                        (sp.matrix_multiply_elementwise((self.position +self.phase_shift), self.multiplicity)).applyfunc(sp.cos)) + self.yOffset
        self.V_orig = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDim - 1))

    def set_degree_mode(self):
        #self.dvdpos = lambda positions: np.squeeze(self._calculate_dVdpos(*np.deg2rad(positions)))
        #self.ene = lambda positions: np.squeeze(self._calculate_energies(*np.deg2rad(positions)))
        self.ene = lambda positions: np.squeeze(self._calculate_energies(*np.hsplit(np.deg2rad(positions), self.constants[self.nDim])))
        self.dvdpos = lambda positions: np.squeeze(self._calculate_dVdpos(*np.hsplit(np.deg2rad(positions), self.constants[self.nDim])))

    def set_radian_mode(self):
        self.ene = lambda positions: np.squeeze(self._calculate_energies(*np.hsplit(positions, self.constants[self.nDim])))
        self.dvdpos = lambda positions: np.squeeze(self._calculate_dVdpos(*np.hsplit(positions, self.constants[self.nDim])))


class torsionPotential(_potential2DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "Torsion Potential"

    phase:float=1.0
    wave_potentials:List[wavePotential]=[]

    def __init__(self, wave_potentials:List[wavePotential]):
        '''
        initializes torsions Potential
        '''
        super().__init__()

        if(isinstance(wave_potentials, Sized) and len(wave_potentials) > 1):
            self.wave_potentials = wave_potentials
        else:
            raise Exception("Torsion Potential needs at least two Wave functions. Otherewise please use wave Potentials.")

    def _calculate_energies_multiPos(self, positions: Iterable[Iterable[Number]]) ->  np.array:
        return np.add(*map(lambda x: np.array(x.ene(positions)), self.wave_potentials))

    def _calculate_energies_singlePos(self, position: Iterable[float]) -> np.array:
        return np.add(*map(lambda x: np.array(x.ene(position)), self.wave_potentials))

    def _calculate_dhdpos_multiPos(self, positions: Iterable[Iterable[float]]) ->  np.array:
        return  np.add(*map(lambda x: np.array(x.dhdpos(positions)), self.wave_potentials))

    def _calculate_dhdpos_singlePos(self, position:Iterable[float]) -> np.array:
        return  np.add(*map(lambda x: np.array(x.dhdpos(position)), self.wave_potentials))

