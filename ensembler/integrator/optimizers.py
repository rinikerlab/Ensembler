"""
 Stochastic Integrators

"""

import numpy as np
from typing import Tuple
import scipy.constants as const
from scipy.optimize import fmin_cg
from ensembler.integrator._basicIntegrators import _integratorCls


class optimizer(_integratorCls):
    pass

class conjugate_gradient(optimizer):
    epsilon:float
    maxStepSize:float

    def __init__(self, max_step_size:float=1, epsilon:float=10 ** -20):
        self.epsilon = epsilon
        self.maxStepSize = max_step_size

    def step(self, system)-> Tuple[float, None, float]:
        """
        ..autofunction: step
            This function is performing an integration step for optimizing a potential.
            NOTE: This is not the optimal way to call scipy optimization, but it works with the interfaces and is useful for theoretical thinking about it. 
            It is not optimal or even efficient here! -> it simulates the optimization process as a stepwise process
        :param system: This is a system, that should be integrated.
        :type system: ensembler.system.system
        :return: (new Position, None, position Shift)
        :rtype: (float, None, float)
        """

        f = system.potential.ene
        f_prime = system.potential.dvdpos

        self.oldpos = system.currentState.position
        self.newPos = np.squeeze(fmin_cg(f=f, fprime=f_prime, x0=self.oldpos, epsilon=self.epsilon, maxiter=1, disp=False))

        move_vector = self.oldpos-self.newPos

        #Scale if move vector is bigger than maxstep
        prefactor=abs(self.maxStepSize/move_vector) if(self.oldpos-self.newPos!=0) else 0
        if(not isinstance(self.maxStepSize, type(None)) and prefactor<1):
            scaled_move_vector = prefactor* move_vector
            self.newPos = self.oldpos-scaled_move_vector
        
        return self.newPos, np.nan, move_vector


    def optimize(self, potential, x0, maxiter = 100)->dict:
        """
        optimize [summary]

        :param potential: [description]
        :type potential: [type]
        :param x0: [description]
        :type x0: [type]
        :param maxiter: [description], defaults to 100
        :type maxiter: int, optional
        :raises ValueError: [description]
        :return: [description]
        :rtype: dict
        """

        cg_out = fmin_cg(f=potential.ene, fprime=potential.dvdpos, x0=x0, epsilon=self.epsilon, maxiter=maxiter,
                         full_output=True,retall=True)
        opt_position, Vmin, function_iterations, gradient_iterations, warn_flag, traj_positions = cg_out

        if(warn_flag==1):
            raise ValueError("Did not converge with the maximal number of iterations")
        elif(warn_flag==2):
            raise ValueError("Function values did not change! Error in function? precision problem?")
        elif(warn_flag==3):
            raise ValueError("encountered NaN")

        return {"optimal_position":opt_position, "minimal_potential_energy":Vmin, "used_iterations":function_iterations, "position_trajectory":traj_positions}

class cg(conjugate_gradient):
    pass
