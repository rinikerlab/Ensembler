"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""
from typing import List

import numpy as np
import sympy as sp

from ensembler.potentials._basicPotentials import _potential2DCls
from ensembler.util.ensemblerTypes import systemCls


class harmonicOscillatorPotential(_potential2DCls):
    """
        Implementation of an 2D  harmonic oscillator potential following hooke's law
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

    def __init__(self, k: np.array = np.array([1.0, 1.0]), r_shift: np.array = np.array([0.0, 0.0]),
                 Voff: np.array = np.array([0.0, 0.0])):
        """
        __init__
            This is the Constructor of the 2D harmonic oscillator

        Parameters
        ----------
        k: array, optional
            force constants in x and y direction, defaults to [1.0, 1.0]
        r_shift: array, optional
            shift of the minimum in the x and y direction, defaults to [0.0, 0.0]
        Voff: array, optional
            offset of the minimum, defaults to [0.0, 0.0]
        """
        self.constants= {self.nDimensions: 2}
        self.constants.update({"k_" + str(j): k[j] for j in range(self.constants[self.nDimensions])})
        self.constants.update({"r_shift" + str(j): r_shift[j] for j in range(self.constants[self.nDimensions])})
        self.constants.update({"V_off_" + str(j): Voff[j] for j in range(self.constants[self.nDimensions])})
        super().__init__()

    def _initialize_functions(self):
        """
        _initialize_functions
            converts the symbolic mathematics of sympy to a matrix representation that is compatible
            with multi-dimentionality.
        """
        # Parameters
        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])
        self.r_shift = sp.Matrix([sp.symbols("r_shift" + str(i)) for i in range(nDimensions)])
        self.V_off = sp.Matrix([sp.symbols("V_off_" + str(i)) for i in range(nDimensions)])
        self.k = sp.Matrix([sp.symbols("k_" + str(i)) for i in range(nDimensions)])
        # Function
        self.V_dim = 0.5 * sp.matrix_multiply_elementwise(self.k, (
            (self.position - self.r_shift).applyfunc(lambda x: x ** 2)))  # +self.Voff
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDimensions - 1))


class wavePotential(_potential2DCls):
    """
    Simple 2D wave potential consisting of cosine functions with given multiplicity, that can be shifted and elongated
    """
    name: str = "Wave Potential"
    nDimensions: sp.Symbol = sp.symbols("nDimensions")

    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    multiplicity: sp.Matrix = sp.Matrix([sp.symbols("m")])
    phase_shift: sp.Matrix = sp.Matrix([sp.symbols("omega")])
    amplitude: sp.Matrix = sp.Matrix([sp.symbols("A")])
    yOffset: sp.Matrix = sp.Matrix([sp.symbols("y_off")])

    V_dim = sp.matrix_multiply_elementwise(amplitude,
                                           (sp.matrix_multiply_elementwise((position + phase_shift),
                                                                           multiplicity)).applyfunc(sp.cos)) + yOffset
    i = sp.Symbol("i")
    V_functional = sp.Sum(V_dim[i, 0], (i, 0, nDimensions))

    def __init__(self, amplitude=(1, 1), multiplicity=(1, 1), phase_shift=(0, 0), y_offset=(0, 0),
                 radians: bool = False):
        """
        __init__
            This is the Constructor of the 2D wave potential function
        Parameters
        ----------
        amplitude: tuple, optional
            absolute min and max of the potential for the cosines in x and y direction, defaults to (1, 1)
        multiplicity: tuple, optional
            amount of minima in one phase for the cosines in x and y direction, defaults to (1, 1)
        phase_shift: tuple, optional
            position shift of the potential for the cosines in x and y direction, defaults to (0, 0)
        y_offset: tuple, optional
            potential shift for the cosines in x and y direction, defaults to (0, 0)
        radians: bool, optional
            in radians or degrees, defaults to False
        """
        self.radians = radians
        nDimensions = 2

        self.constants = {"amp_" + str(j): amplitude[j] for j in range(nDimensions)}
        self.constants.update({"yOff_" + str(j): y_offset[j] for j in range(nDimensions)})
        self.constants.update({"mult_" + str(j): multiplicity[j] for j in range(nDimensions)})

        if(radians):
            self.constants.update({"phase_" + str(j): phase_shift[j] for j in range(nDimensions)})
        else:
            self.constants.update({"phase_" + str(j): np.deg2rad(phase_shift[j]) for j in range(nDimensions)})

        super().__init__()


    def _initialize_functions(self):
        """
        _initialize_functions
            converts the symbolic mathematics of sympy to a matrix representation that is compatible
            with multi-dimentionality.
        """
        # Parameters
        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])
        self.multiplicity = sp.Matrix([sp.symbols("mult_" + str(i)) for i in range(nDimensions)])
        self.phase_shift = sp.Matrix([sp.symbols("phase_" + str(i)) for i in range(nDimensions)])
        self.amplitude = sp.Matrix([sp.symbols("amp_" + str(i)) for i in range(nDimensions)])
        self.yOffset = sp.Matrix([sp.symbols("yOff_" + str(i)) for i in range(nDimensions)])

        # Function
        self.V_dim = sp.matrix_multiply_elementwise(self.amplitude,
                                                    (sp.matrix_multiply_elementwise((self.position + self.phase_shift),
                                                                                    self.multiplicity)).applyfunc(
                                                        sp.cos)) + self.yOffset
        self.V_functional = sp.Sum(self.V_dim[self.i, 0], (self.i, 0, self.nDimensions - 1))

    # OVERRIDE
    def _update_functions(self):
        """
        _update_functions
            calculates the current energy and derivative of the energy
        """
        super()._update_functions()

        self.tmp_Vfunc = self._calculate_energies
        self.tmp_dVdpfunc = self._calculate_dVdpos

        self.set_radians(self.radians)


    def set_phaseshift(self, phaseshift):
        nDimensions = self.constants[self.nDimensions]

        self.constants.update({"phase_" + str(j): phaseshift[j] for j in range(nDimensions)})
        self._update_functions()

    def set_degrees(self, degrees: bool = True):
        """
        Sets output to either degrees or radians

        Parameters
        ----------
        degrees: bool, optional,
            if True, output will be given in degrees, otherwise in radians, default: True
        """
        self.radians = bool(not degrees)
        if (degrees):
            self._calculate_energies = lambda positions, positions2: self.tmp_Vfunc(np.deg2rad(positions),
                                                                                    np.deg2rad(positions2))
            self._calculate_dVdpos = lambda positions, positions2: self.tmp_dVdpfunc(np.deg2rad(positions),
                                                                                     np.deg2rad(positions2))
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
            self.set_degrees(degrees=bool(not radians))


class addedWavePotential(_potential2DCls):
    """
    Adds two wave potentials
    """
    name: str = "Torsion Potential"

    position = sp.symbols("r")
    wave_potentials: sp.Matrix = sp.Matrix([sp.symbols("V_x")])

    nWavePotentials = sp.symbols("N")
    i = sp.symbols("i", cls=sp.Idx)

    V_functional = sp.Sum(wave_potentials[i, 0], (i, 0, nWavePotentials))

    def __init__(self, wave_potentials: List[wavePotential] = (wavePotential(), wavePotential(multiplicity=[3, 3])),
                 degrees: bool = True):
        """
        __init__
            This is the Constructor of an added wave Potential

        Parameters
        ----------
        wavePotentials: list of two 2D potentialTypes, optional
            is uses the 2D wave potential class to generate its potential,
            default to (wavePotential(), wavePotential(multiplicity=[3, 3]))
        radians: bool, optional
            set potential to radians or degrees, defaults to False
        """
        self.constants = {self.nWavePotentials: len(wave_potentials)}
        self.constants.update({"V_" + str(i): wave_potentials[i].V for i in range(len(wave_potentials))})

        super().__init__()
        self.set_degrees(degrees=degrees)

    def _initialize_functions(self):
        """
        _initialize_functions
            converts the symbolic mathematics of sympy to a matrix representation that is compatible
            with multi-dimentionality.
        """
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(self.constants[self.nDimensions])])
        self.wave_potentials = sp.Matrix(
            [sp.symbols("V_" + str(i)) for i in range(self.constants[self.nWavePotentials])])
        # Function
        self.V_functional = sp.Sum(self.wave_potentials[self.i, 0], (self.i, 0, self.nWavePotentials - 1))

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
            self._calculate_energies = lambda positions, positions2: self.tmp_Vfunc(np.deg2rad(positions),
                                                                                    np.deg2rad(positions2))
            self._calculate_dVdpos = lambda positions, positions2: self.tmp_dVdpfunc(np.deg2rad(positions),
                                                                                     np.deg2rad(positions2))
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


class gaussPotential(_potential2DCls):
    '''
        Gaussian like potential, usually used for metadynamics
    '''
    name: str = "Gaussian Potential 2D"
    nDimensions: sp.Symbol = sp.symbols("nDimensions")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    mean: sp.Matrix = sp.Matrix([sp.symbols("mu")])
    sigma: sp.Matrix = sp.Matrix([sp.symbols("sigma")])
    amplitude = sp.symbols("A_gauss")

    # we assume that the two dimentions are uncorrelated
    # V_dim = amplitude * (sp.matrix_multiply_elementwise((position - mean) ** 2, (2 * sigma ** 2) ** (-1)).applyfunc(sp.exp))
    V_dim = amplitude * (sp.matrix_multiply_elementwise(-(position - mean).applyfunc(lambda x: x ** 2),
                                                        0.5 * (sigma).applyfunc(lambda x: x ** (-2))).applyfunc(sp.exp))

    i = sp.Symbol("i")
    V_functional = sp.summation(V_dim[i, 0], (i, 0, nDimensions))

    # V_orig = V_dim[0, 0] * V_dim[1, 0]

    def __init__(self, amplitude=1., mu=(0., 0.), sigma=(1., 1.), negative_sign:bool=False):
        '''
         __init__
            This is the Constructor of a 2D Gauss Potential

        Parameters
        ----------
        A: float, optional
            scaling of the gauss function, defaults to 1.
        mu: tupel, optional
            mean of the gauss function, defaults to (0., 0.)
        sigma: tupel, optional
            standard deviation of the gauss function, defaults to (1., 1.)
        negative_sign: bool, optional
            this option is switching the sign of the final potential energy landscape. ==> mu defines the minima location, not maxima location
        '''


        nDimensions = 2
        self.constants= {"A_gauss": amplitude}
        self.constants.update({"mu_" + str(j): mu[j] for j in range(nDimensions)})
        self.constants.update({"sigma_" + str(j): sigma[j] for j in range(nDimensions)})
        self.constants.update({self.nDimensions:nDimensions})
        self._negative_sign=negative_sign
        super().__init__()

    def _initialize_functions(self):
        """
        _initialize_functions
            converts the symbolic mathematics of sympy to a matrix representation that is compatible
            with multi-dimentionality.
        """
        # Parameters
        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])
        self.mean = sp.Matrix([sp.symbols("mu_" + str(i)) for i in range(nDimensions)])
        self.sigma = sp.Matrix([sp.symbols("sigma_" + str(i)) for i in range(nDimensions)])
        self.amplitude = sp.symbols("A_gauss")

        # Function
        self.V_dim = self.amplitude * (
            sp.matrix_multiply_elementwise(-(self.position - self.mean).applyfunc(lambda x: x ** 2),
                                           0.5 * (self.sigma).applyfunc(lambda x: x ** (-2))).applyfunc(sp.exp))

        # self.V_functional = sp.Product(self.V_dim[self.i, 0], (self.i, 0, self.nDimensions- 1))
        # Not too beautiful, but sp.Product raises errors
        if(self._negative_sign):
            self.V_functional = -(self.V_dim[0, 0] * self.V_dim[1, 0])
        else:
            self.V_functional = self.V_dim[0, 0] * self.V_dim[1, 0]

    def _update_functions(self):
        """
        This function is needed to simplyfiy the symbolic equation on the fly and to calculate the position derivateive.
        """

        self.V = self.V_functional.subs(self.constants)

        self.dVdpos_functional = sp.diff(self.V_functional, self.position)  # not always working!
        self.dVdpos = sp.diff(self.V, self.position)
        self.dVdpos = self.dVdpos.subs(self.constants)

        self._calculate_energies = sp.lambdify(self.position, self.V, "numpy")
        self._calculate_dVdpos = sp.lambdify(self.position, self.dVdpos, "numpy")


from ensembler.potentials.ND import envelopedPotential, sumPotentials

"""
Biased potentials
"""
"""
    TIME INDEPENDENT BIASES 
"""


class addedPotentials(_potential2DCls):
    '''
    Adds two different potentials on top of each other. Can be used to generate
    harmonic potential umbrella sampling or scaled potentials
    '''

    name: str = "Added Potential Enhanced Sampling System for 2D"
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])
    bias_potential = True

    def __init__(self, origPotential=harmonicOscillatorPotential(), addPotential=gaussPotential()):
        '''
        __init__
              This is the Constructor of the addedPotential class.
        Parameters
        ----------
        origPotential: 2D potential type
            The unbiased potential
        addPotential: 2D potential type
            The potential added on top of the unbiased potential to
            bias the system
        '''

        self.origPotential = origPotential
        self.addPotential = addPotential

        self.constants = {**origPotential.constants, **addPotential.constants}

        self.V_functional = self.origPotential.V + self.addPotential.V

        self.V = self.V_functional.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

    def _initialize_functions(self):
        # Parameters
        print(self.nDimensions)

        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])

"""
    TIME DEPENDENT BIASES 
"""


class metadynamicsPotential(_potential2DCls):
    '''
    The metadynamics bias potential adds 2D Gaussian potentials on top of
    the original 2D potential. The added gaussian potential is centered on the current position.
    Thereby the valleys of the potential "flooded" and barrier crossing is easier.

    This implementation uses a grid to store the biasing. This is much faster than calculating
    an ever increasing potential with sympy
    '''

    name: str = "Metadynamics Enhanced Sampling System using grid bias in 2D"
    position = sp.symbols("r")
    system: systemCls  # metadyn-coupled to system
    bias_potential = True

    def __init__(self, origPotential=harmonicOscillatorPotential(), amplitude=1., sigma=(1., 1.), n_trigger=100,
                 bias_grid_min=(0, 0),
                 bias_grid_max=(10, 10), numbins=(100, 100)):
        '''

        Parameters
        ----------
        origPotential: potential 2D type
            The unbiased potential
        amplitude: float
            scaling of the gaussian potential added in the metadynamcis step
        sigma:tuple
            standard deviation of the gaussian potential in x and y added in the metadynamcis step
        n_trigger: int
            Metadynamics potential will be added after every n_trigger'th steps
        bias_grid_min:tuple
            min value in x and y direction for the grid
        bias_grid_max:tuple
            max value in x and y direction for the grid
        numbins: tuple
            size of the grid bias and forces are saved in
        '''
        self.origPotential = origPotential
        self.n_trigger = n_trigger
        self.amplitude = amplitude
        self.sigma = sigma
        # grid where the bias is stored
        # for 2D
        self.numbins = numbins
        self.shape_force = (2, numbins[0], numbins[1])  # 2 because 2D
        self.bias_grid_energy = np.zeros(self.numbins)  # energy grid
        self.bias_grid_force = np.zeros(self.shape_force)  # force grid

        # get center value for each bin
        bin_half_x = (bias_grid_max[0] - bias_grid_min[0]) / (2 * self.numbins[0])  # half bin width
        bin_half_y = (bias_grid_max[1] - bias_grid_min[1]) / (2 * self.numbins[1])  # half bin width
        self.x_centers = np.linspace(bias_grid_min[0] + bin_half_x, bias_grid_max[0] - bin_half_x, self.numbins[0])
        self.y_centers = np.linspace(bias_grid_min[1] + bin_half_y, bias_grid_max[0] - bin_half_y, self.numbins[1])
        self.bin_centers = np.array(np.meshgrid(self.x_centers, self.y_centers))
        self.positions_grid = np.array([self.bin_centers[0].flatten(), self.bin_centers[1].flatten()]).T

        # current_n counts when next metadynamic step should be applied
        self.current_n = 1

        self.constants = {**origPotential.constants}

        self.V_functional = origPotential.V
        self.V_orig_part = self.V_functional.subs(self.constants)

        super().__init__()

    def _initialize_functions(self):
        # Parameters
        nDimensions = self.constants[self.nDimensions]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDimensions)])

    '''
    BIAS
    '''

    # Beautiful integration to system as Condition.
    def apply(self):
        self.check_for_metastep(self.system._currentPosition)

    def apply_coupled(self):
        self.check_for_metastep(self.system._currentPosition)

    def couple_system(self, system):
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

    def _update_potential(self, curr_position):
        '''
        Is triggered by check_for_metastep(). Adds a gaussian centered on the
        current position to the potential

        Parameters
        ----------
        curr_position: tuple
            current x,y position

        Returns
        -------
        '''
        # do gaussian metadynamics
        new_bias = gaussPotential(amplitude=self.amplitude, mu=curr_position, sigma=self.sigma)

        # size energy and force of the new bias in bin structure
        new_bias_lambda_energy = new_bias._calculate_energies #sp.lambdify(self.position, new_bias.V)
        new_bias_lambda_force = new_bias._calculate_dVdpos #sp.lambdify(self.position, new_bias.dVdpos)

        new_bias_bin_energy = new_bias_lambda_energy(*np.hsplit(self.positions_grid, self.constants[self.nDimensions]))
        new_bias_bin_force = new_bias_lambda_force(*np.hsplit(self.positions_grid, self.constants[self.nDimensions]))
        # update bias grid
        self.bias_grid_energy = self.bias_grid_energy + new_bias_bin_energy.reshape(self.numbins)

        self.bias_grid_force = self.bias_grid_force + new_bias_bin_force.reshape(self.shape_force)

    # overwrite the energy and force
    def ene(self, positions):
        '''
        calculates energy of particle also takes bias into account
        Parameters
        ----------
        positions: tuple
            position on 2D potential energy surface

        Returns
        -------
        current energy
        '''
        current_bin_x = self._find_nearest(self.x_centers, np.array(np.array(positions, ndmin=1).T[0], ndmin=1))
        current_bin_y = self._find_nearest(self.y_centers, np.array(np.array(positions, ndmin=1).T[1], ndmin=1))
        enes = np.squeeze(
            self._calculate_energies(*np.hsplit(np.array(positions, ndmin=1), self.constants[self.nDimensions])))
        biases = self.bias_grid_energy[current_bin_y, current_bin_x]
        return np.squeeze(enes + biases)

    def force(self, positions):
        '''
        calculates derivative with respect to position also takes bias into account

        Parameters
        ----------
        positions: tuple
            position on 2D potential energy surface

        Returns
        current derivative dh/dpos
        -------
        '''

        x_vals = np.array(np.array(positions, ndmin=1).T[0], ndmin=1)
        y_vals = np.array(np.array(positions, ndmin=1).T[1], ndmin=1)
        current_bin_x = self._find_nearest(self.y_centers, x_vals)
        current_bin_y = self._find_nearest(self.y_centers, y_vals)

        arr = np.array(positions, ndmin=2) #Check is this a good solution?
        f = [self._calculate_dVdpos(x,y) for x,y in arr]
        dvdpos = np.squeeze(f)

        return np.squeeze(dvdpos + self.bias_grid_force[:, current_bin_y, current_bin_x].T)

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

        centers = []
        for val in value:
            idx = np.searchsorted(array, val, side="left")
            if idx > 0 and (idx == len(array) or np.abs(val - array[idx - 1]) < np.abs(val - array[idx])):
                centers.append(idx - 1)
            else:
                centers.append(idx)
        return np.array(centers, ndmin=1)
