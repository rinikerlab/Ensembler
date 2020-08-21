"""
Module: Potential
This module shall be used to implement biases on top of Potentials. This module contains all available 2 Biases.

"""

import numpy as np
import sympy as sp

from ensembler.potentials.TwoD import gaussPotential
from ensembler.potentials._basicPotentials import _potential2DCls
from ensembler.util.ensemblerTypes import system

"""
    TIME INDEPENDENT BIASES 
"""


class addedPotentials(_potential2DCls):
    '''
    Adds two different potentials on top of each other. Can be used to generate
    harmonic potential umbrella sampling or scaled potentials
    '''

    name: str = "Added Potential Enhanced Sampling System for 2D"
    nDim: int = sp.symbols("nDim")
    position: sp.Matrix = sp.Matrix([sp.symbols("r")])

    def __init__(self, origPotential, addPotential):
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

        self.V_orig = origPotential.V + self.addPotential.V

        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDim)])


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
    system: system  # metadyn-coupled to system

    def __init__(self, origPotential, amplitude=1., sigma=(1., 1.), n_trigger=100, bias_grid_min=(0, 0),
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

        self.V_orig = origPotential.V
        self.V_orig_part = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V_orig_part, self.position)
        self.V = self.V_orig_part

        super().__init__()

    def _initialize_functions(self):
        # Parameters
        nDim = self.constants[self.nDim]
        self.position = sp.Matrix([sp.symbols("r_" + str(i)) for i in range(nDim)])

    """
    BIAS
    """

    # Beautiful integration to system as Condition.
    def apply(self):
        self.check_for_metastep(self.system._currentPosition)

    def coupleSystem(self, system):
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
        curr_position: tuple
            current x,y position

        Returns
        -------
        '''
        # do gaussian metadynamics
        new_bias = gaussPotential(amplitude=self.amplitude, mu=curr_position, sigma=self.sigma)

        # size energy and force of the new bias in bin structure
        new_bias_lambda_energy = sp.lambdify(self.position, new_bias.V)
        new_bias_lambda_force = sp.lambdify(self.position, new_bias.dVdpos)

        new_bias_bin_energy = new_bias_lambda_energy(*np.hsplit(self.positions_grid, self.constants[self.nDim]))
        new_bias_bin_force = new_bias_lambda_force(*np.hsplit(self.positions_grid, self.constants[self.nDim]))
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
        current_bin_x = self._find_nearest(self.x_centers, positions[0])
        current_bin_y = self._find_nearest(self.y_centers, positions[1])
        # due to transposed position matrix, x and y are changed here
        return np.squeeze(
            self._calculate_energies(*np.hsplit(positions, self.constants[self.nDim])) + self.bias_grid_energy[
                current_bin_y, current_bin_x])

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

        current_bin_x = self._find_nearest(self.x_centers, positions[0])
        current_bin_y = self._find_nearest(self.y_centers, positions[1])
        # due to transposed position matrix, x and y are changed here
        return np.squeeze(
            self._calculate_dVdpos(*np.hsplit(positions, self.constants[self.nDim])) + self.bias_grid_force[:,
                                                                                       current_bin_y,
                                                                                       current_bin_x].reshape(2, 1, 1))

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
