"""
Module: Potential
This module shall be used to implement biases on top of Potentials. This module contains all available biases.
"""


import numpy as np
import sympy as sp


from ensembler.potentials._baseclasses import _potential1DClsSymPY

"""
    BIAS BASECLASS
"""


class _bias_baseclass(_potential1DClsSymPY):

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

        self.constants = {self.y_shift: y_shift}
        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()

        self._calculate_energies = lambda positions: np.squeeze(np.full(len(positions), y_shift))
        self.dVdpos = self._calculate_dVdpos = lambda positions: np.squeeze(np.zeros(len(positions)))

    def return_bias_action(self):

        # important for reweighting
        #check if integrator is Langevin
        if isinstance(self.integrator , langevinIntegrator):
            # calculate path action biased
            # todo: check how to best use the R_x for biased and unbiased
            # --> simulation potential has to be
            P_x = (1 / (np.sqrt(2 * np.pi) * np.sqrt(2 * self.system.temperature * self.integrator.gamma * self.system.mass))) * np.exp(
                -(self.integrator.R_x ** 2) / (2 * (2 * self.system.temperature * self.integrator.gamma * self.system.mass)))
            return -np.log(P_x)
        # if not raise error
        else:
            raise TypeError("Integrator has to be of type langevinIntegrator to calculate the actions")


    def return_target_action(self):
        # important for reweighting

        # only works with Langevin
        Pass

    def return_bias_energy(self):
        # important for reweighting

        Pass

    def return_path_correction_term(self):
        # important for reweighting

        #works with newtonion or langevin Integrator

        #pass


        #get reweighting parameter necessary for Weber reweighting
        #:return: bias structure

        # calculate new force and corresponding random number
        F_rw = -system.potential.unbiased_system_dhdpos(currentPosition)  # has to be defined
        R_rw = (self.newForces - F_rw) + self.R_x

        # calculate path action unbiased
        P_x_rw = (1 / (np.sqrt(2 * np.pi) * np.sqrt(2 * system.temperature * self.gamma * system.mass))) * np.exp(
            -(R_rw ** 2) / (2 * (2 * system.temperature * self.gamma * system.mass)))
        A_rw = -np.log(P_x_rw)

        #get action difference for this step
        A_diff = A - A_rw

        return A_diff
        
        

"""
    TIME INDEPENDENT BIASES 
"""

class addedPotentials(_potential1DClsSymPY):

    name:str = "Added Potential Enhanced Sampling System"
    position = sp.symbols("r")

    def __init__(self, origPotential, addPotential):

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

        self.origPotential  = origPotential
        self.addPotential = addPotential

        self.constants = {**origPotential.constants, **addPotential.constants}

        self.V_orig =  origPotential.V + addPotential.V 

        self.V = self.V_orig.subs(self.constants)
        self.dVdpos = sp.diff(self.V, self.position)

        super().__init__()



"""
    TIME DEPENDENT BIASES 
"""
# include metadynamics

