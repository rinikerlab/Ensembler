"""
Module: Potential
This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import math
import numpy as np
import sympy as sp
from numbers import Number
from typing import Iterable, Sized, Union
import typing as t
from numbers import Number
import scipy.constants as const
from collections.abc import Iterable, Sized

from ensembler.potentials import ND
from ensembler.potentials._baseclasses import _potential1DCls, _perturbedPotentialNDCls, _potential1DClsSymPY, _potential1DClsSymPYPerturbed

"""
    SIMPLE POTENTIALS
"""

class dummyPotential(_potential1DClsSymPY):

    name:str = "dummyPotential"
    position, y_shift = sp.symbols("r Voffset")

    def __init__(self, y_shift: float = 0):
        """
        This Class is representing the a dummy potential. 
        It returns a constant value equalling the y_shift parameter.
        
        :param y_shift: This will be the constant return value, defaults to 0
        :type y_shift: float, optional
        """

        self.V_orig = sp.Lambda(self.position, self.y_shift)

        self.constants = {self.y_shift:y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)
        super().__init__()
        
        self._calculate_energies = lambda positions: np.squeeze(np.full(len(positions), y_shift))
        self.dVdpos = self._calculate_dVdpos = lambda positions: np.squeeze(np.zeros(len(positions)))

class flatwell(_potential1DCls):
    name:str = "Flat Well"

    x_min: float = None
    x_max: float = None
    y_max:float = None
    y_min:float = None

    def __init__(self, x_range: list = (0, 1), y_max: float = 1000, y_min: float = 0):
        """
        __init__ This potential is a flatwell potential. 
        The flatwell potential is a function similar to an if case.
        If a position is inside a the x_range, it returns the y_min val.
        If a position is outside, the y_max val will be returned.

        :param x_range: range inside this the y_min val will be returned, defaults to (0, 1)
        :type x_range: list, optional
        :param y_max: outside of the range this value will be returned, defaults to 1000
        :type y_max: float, optional
        :param y_min: inside the range this value will be returned, defaults to 0
        :type y_min: float, optional
        """
        super().__init__()
        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.y_min if (position >= self.x_min and position <= self.x_max) else self.y_max

    def _calculate_dvdpos_singlePos(self, positions: float) -> float:
        return np.inf

class harmonicOscillator(_potential1DClsSymPY):
    name:str = "harmonicOscilator"
    k, x_shift, position, y_shift = sp.symbols("k r_0 r Voffset")
    V_orig = 0.5*k*(position-x_shift)**2+y_shift

    def __init__(self, k: float = 1.0, x_shift: float = 0.0, y_shift: float = 0.0):
        """
        implementation of an Harmonic Oscilator following hooke's law
        
        :param k: force constant, defaults to 1.0
        :type k: float, optional
        :param x_shift: shift on the x Axis, defaults to 0.0
        :type x_shift: float, optional
        :param y_shift: shift on the y Axis, defaults to 0.0
        :type y_shift: float, optional
        """
        self.constants = {self.k:k, self.x_shift:x_shift, self.y_shift:y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

class wavePotential(_potential1DClsSymPY):
    name:str = "Wave Potential"
    amplitude, phase_shift, position, y_shift, multiplicity = sp.symbols("A w r Voff m")
    V_orig = amplitude * sp.cos(multiplicity * (position + phase_shift)) + y_shift

    def __init__(self, amplitude: float = 1.0,  multiplicity:float =1.0, phase_shift: float = 0.0, y_shift: float = 0.0, radians: bool = False):
        """
        Implemenation of a wave function.
        
        :param amplitude: absolute min and max of the potential, defaults to 1.0
        :type amplitude: float, optional
        :param multiplicity: ammount of minima in one phase, defaults to 1.0
        :type multiplicity: float, optional
        :param phase_shift: shift of the potential on the x Axis, defaults to 0.0
        :type phase_shift: float, optional
        :param y_shift: shift on the y Axis, defaults to 0.0
        :type y_shift: float, optional
        :param radians: in radians or degrees - NOT IMPLEMNTED!, defaults to False
        :type radians: bool, optional
        """

        self.constants = {self.amplitude:amplitude, self.multiplicity:multiplicity,  self.phase_shift:phase_shift, self.y_shift:y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()
        
        #self.set_radians(radians)

    """
    def set_degrees(self, degrees: bool = True):
        if(degrees):
            self.tmp_Vfunc = self._calculate_energies
            self.tmp_dVdpfunc = self._calculate_dVdpos

            self._calculate_energies = lambda positions: self.tmp_Vfunc(np.deg2rad(positions))
            self._calculate_dVdpos = lambda positions: self.tmp_dVdpfunc(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            self._calculate_energies  =self.tmp_Vfunc
            self._calculate_dVdpos = self.tmp_dVdpfunc
        else:
            self.set_degrees(degrees=not radians)
    """
     
class coulombPotential(_potential1DClsSymPY):
    name = "Coulomb Potential"
    charge1, charge2, position, electric_permetivity = sp.symbols("q1 q2 r e")
    V_orig = (charge1 * charge2)/(position * electric_permetivity * 4 * sp.pi)

    def __init__(self, q1=1, q2=1, epsilon=1):
        """
        Implemenation of Coulomb Law
        
        :param q1: Charge of atom 1, defaults to 1
        :type q1: int, optional
        :param q2: Charge of atom 2, defaults to 1
        :type q2: int, optional
        :param epsilon: Electric Permetivitty, defaults to 1
        :type epsilon: int, optional
        """
        self.constants = {self.charge1:q1, self.charge2:q2, self.electric_permetivity:epsilon}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

class LennardJonesPotential(_potential1DClsSymPY):
    name:str = "Lennard Jones Potential"
    sigma, epsilon, x_shift, y_shift, position = sp.symbols("s e r_0 V_off r")
    V_orig = 4 * epsilon * ((sigma/(position-x_shift))**12 - (sigma/(position-x_shift))**6) + y_shift 

    def __init__(self, sigma: float = 1.5, epsilon: float = 2, x_shift: float = 0, y_shift=0):
        """
        Implementation of lennard Jones Implementation.
        
        :param sigma: x - Position of the minimum, defaults to 1.5
        :type sigma: float, optional
        :param epsilon: y position of minima, defaults to 2
        :type epsilon: float, optional
        :param x_shift: movement of potential on x Axis, defaults to 0
        :type x_shift: float, optional
        :param y_shift: movement of potential on y Axis, defaults to 0
        :type y_shift: int, optional
        """
        self.constants = {self.sigma:sigma, self.epsilon:epsilon, self.x_shift:x_shift, self.y_shift:y_shift}

        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

class lennardJonesForceFieldPotential(_potential1DClsSymPY):

    name:str = "Lennard Jones Potential"
    c6, c12, x_shift, y_shift, position = sp.symbols("c6 c12 r_0 V_off r")
    V_orig = (c12/(position-x_shift)**12)-(c6/(position-x_shift**6))+y_shift

    def __init__(self, c6: float = 0.2, c12: float = 0.0001, x_shift: float = 0, y_shift:float=0):
        """
        This is a forcefield like implementation of  a lennard Jones Potential
        
        :param c6: [description], defaults to 0.2
        :type c6: float, optional
        :param c12: [description], defaults to 0.0001
        :type c12: float, optional
        :param x_shift: [description], defaults to 0
        :type x_shift: float, optional
        :param y_shift: [description], defaults to 0
        :type y_shift: float, optional
        """
        self.constants = {self.c6:c6, self.c12:c12, self.x_shift:x_shift, self.y_shift:y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

class doubleWellPot(_potential1DClsSymPY):
    name:str = "Double Well Potential"
    a, b, Vmax, position = sp.symbols("a b V_max r")
    V_orig = (Vmax/(b**4)) * ((position - a/2)**2 - b**2) **2

    def __init__(self, Vmax=5, a=-1, b=1):
        """
        This is an implementation of a double Well potential
        
        :param Vmax: Maximal barrier between minima, defaults to 5
        :type Vmax: int, optional
        :param a: [description], defaults to -1
        :type a: int, optional
        :param b: [description], defaults to 1
        :type b: int, optional
        """
        
        self.constants = {self.Vmax:Vmax, self.a:a, self.b:b}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()


"""
    COMBINED POTENTIALS
"""
class torsionPotential(_potential1DClsSymPY):

    name:str = "Torsion Potential"

    phase:float = 1.0
    position = sp.symbols("r")
    wavePotentials = sp.Array([1])
    i, N = sp.symbols("i N")    #sum symbols
    V_orig = sp.Sum(wavePotentials[i,0], (i, 0, N))

    def __init__(self, wavePotentials):
        '''
        initializes torsions Potential
        '''
        self.constants = {**{"wave_"+str(key): wave.V for key, wave in enumerate(wavePotentials)}, **{self.N:len(wavePotentials)-1}}
        self.wavePotentials = sp.Matrix([sp.symbols("wave_" + str(i)) for i in range(len(wavePotentials))])

        self.V_orig = sp.Sum(self.wavePotentials[self.i,0], (self.i, 0,self.N))
        self.V = self.V_orig.subs(self.constants).subs(self.N, len(wavePotentials))

        super().__init__()

class forceField:

    def __init__(self):
        raise NotImplementedError("Not implemented yet, but this calss shall be used to link potential terms! ") 


"""
    PERTURBED POTENTIALS
"""
class linearCoupledPotentials(_potential1DClsSymPYPerturbed):
    name:str = "linear Coupled System"
    lam, position = sp.symbols(u'λ r')
    Va, Vb = (sp.Function("V_a"), sp.Function("V_b"))
    Coupling = (1-lam) * Va(position) + lam * Vb(position)
    
    def __init__(self, Va: _potential1DClsSymPY = harmonicOscillator(k=1.0, x_shift=0.0),
                       Vb: _potential1DClsSymPY = harmonicOscillator(k=11.0, x_shift=0.0), 
                       lam:float = 0.5):
        """
        Linear Coupled Potentials, like in FEP or TI simulations.
        
        :param Va: Potential A, defaults to harmonicOsc(k=1.0, x_shift=0.0)
        :type Va: _potential1DClsSymPY, optional
        :param Vb: Potential B, defaults to harmonicOsc(k=11.0, x_shift=0.0)
        :type Vb: _potential1DClsSymPY, optional
        :param lam: Coupling factor, defaults to 0.5
        :type lam: float, optional
        """
        self.statePotentials =  {self.Va:Va, self.Vb:Vb}
        self.constants = {self.Va:Va.V, self.Vb:Vb.V, self.lam: lam}
        self.V_orig = self.Coupling
        super().__init__()



class exponentialCoupledPotentials(_potential1DClsSymPYPerturbed):
    name:str = "linear Coupled System"
    lam, position, s, temp = sp.symbols(u'λ r s T')
    Va, Vb = (sp.Function("V_a"), sp.Function("V_b"))
    Coupling = (1-lam) * Va(position) + lam * Vb(position)
    beta = const.gas_constant / 1000.0 * temp
    Coupling = -1/(beta*s) * sp.log(lam * sp.exp(-beta*s*Vb(position)) + (1-lam) * sp.exp(-beta*s*Va(position)))

    def __init__(self, Va: _potential1DClsSymPY = harmonicOscillator(k=1.0, x_shift=0.0),
                       Vb: _potential1DClsSymPY = harmonicOscillator(k=11.0, x_shift=0.0), 
                       lam:float = 0.5, s:float = 1.0, temp:float = 298):
        """
        exponential Coupled Potentials, this is a mixture of EDS and TI
    
        :param Va: Potential A, defaults to harmonicOsc(k=1.0, x_shift=0.0)
        :type Va: _potential1DClsSymPY, optional
        :param Vb: Potential B, defaults to harmonicOsc(k=11.0, x_shift=0.0)
        :type Vb: _potential1DClsSymPY, optional
        :param lam: Coupling factor, defaults to 0.5
        :type lam: float, optional
        :param s: smoothing factor, defaults to 1.0
        :type s: float, optional
        :param temp: Temperature, defaults to 298
        :type temp: float, optional
        """

        self.statePotentials =  {self.Va:Va, self.Vb:Vb}
        self.constants = {self.Va:Va.V, self.Vb:Vb.V, self.lam:lam, self.s:s, self.temp:temp}
        
        super().__init__()

    def set_s(self, s:float):
        self.constants.update({self.s:s})
        self._update_functions()

    def set_Eoff(self, Eoff:float):
        self.constants.update({self.Eoff:Eoff})
        self._update_functions()

"""
    ENVELOPED POTENTIALS
    Here old implementations are way more reasonable, otherwise overkill in exponent!
"""
class envelopedPotential(ND.envelopedPotential):
    name = "Enveloping Potential"

    V_is:t.List[_potential1DCls] = None
    E_is:t.List[float] = None
    s:float = None
    nStates:int = None

    kb=const.gas_constant 
    s_s, T, position = sp.symbols("s T r")
    beta = 1/kb*T
    
    Vis = sp.Matrix([1])
    Eoffis = sp.Matrix([1])
    i, N = sp.symbols("i N")
    V_orig = -1/(beta*s_s) * sp.log(sp.Sum(sp.exp(-beta * s_s * (Vis[i,0]-Eoffis[i, 0])),(i, 0, N)))


    def __init__(self, V_is: t.List[_potential1DCls], s: float = 1.0, Eoff_i: t.List[float] = None, T:float=298):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        self.constants.update({self.nDim: 1})
        super().__init__(V_is=V_is, s=s, Eoff_i=Eoff_i)


        self.V_orig = -1/(self.beta*self.s_s) * sp.log(sp.Sum(sp.exp(-self.beta * self.s_s * (self.states[self.i,0])),(self.i, 0, self.N-1)))
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)


    def _calculate_dvdpos_singlePos(self, position:(t.Iterable[float])) -> np.array:
        """
        :warning : Implementation is not entirly correct!
        :param positions:
        :return:
        """
        ###CHECK!THIS FUNC!!! not correct
        V_R_ene = self.ene(position)
        V_Is_ene = np.array([statePot.ene(state_pos) for statePot, state_pos in zip(self.V_is, position)])
        V_Is_dhdpos = np.array([statePot.dvdpos(state_pos) for statePot, state_pos in zip(self.V_is, position)])
        dhdpos = []


        #print("POS: " , position.shape,"\n\t", position,)
        #print("ene: ", V_Is_ene.shape,"\n\t", V_Is_ene)
        #print("dhdpos: ", V_Is_dhdpos.shape,"\n\t", V_Is_dhdpos)
        #print("T", V_Is_ene.T)
        V_Is_posDim_eneSum = np.sum(V_Is_ene.T).T
        #print("sums: ", V_Is_posDim_eneSum.shape, "\n\t", V_Is_posDim_eneSum)

        #prefactors = np.array([np.zeros(len(positions[0])) for x in range(len(positions))])
        #todo: error this should be ref pot fun not sum of all pots
        #FIX FROM HERE ON

        prefactors = np.array([list(map(lambda pos, posSum: list(map(lambda dimPos, dimPosSum: 1 - np.divide(dimPos, dimPosSum), pos, posSum)), Vn_ene, V_Is_posDim_eneSum)) for Vn_ene in V_Is_ene])
        ##print("preFactors: ",prefactors.shape, "\n\t", prefactors,  "\n\t", prefactors.T)
        dhdpos_state_scaled = np.multiply(prefactors, V_Is_dhdpos)
        #print("dhdpos_scaled", dhdpos_state_scaled.shape, "\n\t", dhdpos_state_scaled, "\n\t", dhdpos_state_scaled.T    )

        #dhdpos_R = [  for dhdpos_state in dhdpos_state_scaled]

        dhdpos_R = []
        for position in range(len(position[0])):
            dhdposR_position = []
            for dimPos in range(len(position[0][0])):
                dhdposR_positionDim = 0
                for state in range(len(V_Is_ene)):
                    dhdposR_positionDim = np.add(dhdposR_positionDim, dhdpos_state_scaled[state, position, dimPos])
                dlist = [dhdposR_positionDim]
                dlist.extend(dhdpos_state_scaled[:, position, dimPos])
                dhdposR_position.append(dlist)
            dhdpos_R.append(dhdposR_position)

        return  dhdpos_R.item() if(len(dhdpos_R.shape) == 1 and dhdpos_R.shape[0] == 1) else np.array(dhdpos_R) 
