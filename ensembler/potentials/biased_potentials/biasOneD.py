"""
Module: Potential
This module shall be used to implement biases on top of Potentials. This module contains all available 1D biases.
"""

import numpy as np
import sympy as sp

from ensembler.potentials.OneD import gaussPotential, harmonicOscillatorPotential
from ensembler.potentials._basicPotentials import _potential1DCls

from ensembler.samplers.stochastic import langevinIntegrator
"""
    BIAS BASECLASS
"""


class _bias_baseclass(_potential1DCls):
    name: str = "bias_baseclass"
    nDim = 1
    position, y_shift = sp.symbols("r Voffset")

    def __init__(self, integrator, system: float = 0):
        """
        This Class is representing the a dummy potential.
        It returns a constant value equalling the y_shift parameter.

        :param y_shift: This will be the constant return value, defaults to 0
        :type y_shift: float, optional
        """

        self.integrator = integrator
        self.system = system

        self.V_orig = sp.Lambda(self.position, self.y_shift)

        self.constants = {}#{self.y_shift: y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

        self._calculate_energies = lambda positions: np.squeeze(np.full(len(positions), self.y_shift))
        self.dVdpos = self._calculate_dVdpos = lambda positions: np.squeeze(np.zeros(len(positions)))

    def return_bias_action(self):

        # important for reweighting
        # check if samplers is Langevin
        if isinstance(self.integrator, langevinIntegrator):
            # calculate path action biased
            # todo: check how to best use the R_x for biased and unbiased
            # --> simulation potential has to be
            P_x = (1 / (np.sqrt(2 * np.pi) * np.sqrt(
                2 * self.system.temperature * self.integrator.gamma * self.system.mass))) * np.exp(
                -(self.integrator.R_x ** 2) / (
                            2 * (2 * self.system.temperature * self.integrator.gamma * self.system.mass)))
            return -np.log(P_x)
        # if not raise error
        else:
            raise TypeError("Integrator has to be of type langevinIntegrator to calculate the actions")

    def return_target_action(self):
        # important for reweighting

        # only works with Langevin
        pass

    def return_bias_energy(self):
        # important for reweighting

        pass

    def return_path_correction_term(self):
        # important for reweighting

        # works with newtonion or langevin Integrator

        # pass

        # get reweighting parameter necessary for Weber reweighting
        #:return: bias structure

        # calculate new force and corresponding random number
        F_rw = -self.system.potential.unbiased_system_dhdpos(self.currentPosition)  # has to be defined
        R_rw = (self.newForces - F_rw) + self.R_x

        # calculate path action unbiased
        P_x_rw = (1 / (np.sqrt(2 * np.pi) * np.sqrt(2 * self.system.temperature * self.gamma * self.system.mass))) * np.exp(
            -(R_rw ** 2) / (2 * (2 * self.system.temperature * self.gamma * self.system.mass)))
        A_rw = -np.log(P_x_rw)

        # get action difference for this step
        A_diff = self.A - A_rw

        return A_diff


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

    def __init__(self, origPotential=harmonicOscillatorPotential(), amplitude=0.1, sigma=1, n_trigger=100, bias_grid_min=0, bias_grid_max=10,
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
        #print("A ", self.amplitude, "mu ", curr_position, "sigma ", self.sigma)
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

        print(positions)
        current_bin = self._find_nearest(self.bin_centers, positions)
        force= np.squeeze(self._calculate_dVdpos(np.squeeze(positions)) + self.bias_grid_force[current_bin])
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

        self.constants={}
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
