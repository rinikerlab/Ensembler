"""
Module: Sampler
    The sampler module is provides methods exploring the potential functions.

    Optimization Methods
"""

import numpy as np
from scipy.optimize import fmin_cg

from ensembler.samplers._basicSamplers import _samplerCls
from ensembler.util.ensemblerTypes import systemCls as systemType, Tuple


class optimizer(_samplerCls):
    """
    This class is the parent class for all optimizers. The pre-implemented
    optimizers currently comprise the conjugate gradient.
    """

    maxStepSize: float


class conjugate_gradient(optimizer):
    """
    Conjugate gradient is an algorithm for the numerical solution of linear equations and for energy minimization.
    Linear equations should have a symmetric matrix and be positive-definite.
    """

    epsilon: float

    def __init__(self, max_step_size: float = 1, epsilon: float = 10**-20):
        """
        __init__
            This is the Constructor of the conjugate gradient optimizer

        Parameters
        ----------
        max_step_size : float, optional
            maximal size of an optimization step in any direction, by default 1
        epsilon : float, optional
            Step size(s) to use when the gradient is approximated numerically. Defaults to sqrt(eps),
            with eps the floating point machine precision. Usually sqrt(eps) is about 1.5e-8.,
            by default 10**-20
        """
        super().__init__()

        self.epsilon = epsilon
        self.maxStepSize = max_step_size

    def step(self, system: systemType) -> Tuple[float, None, float]:
        """
        step
            This function is performing an integration step for optimizing a potential.
            NOTE: This is not the optimal way to call scipy optimization, but it works with the interfaces and is useful
            for theoretical thinking about it.
            It is not optimal or even efficient here! -> it simulates the optimization process as a stepwise process

        Parameters
        ----------
        system : systemType
            This is a system, that should be integrated.

        Returns
        -------
        Tuple[float, None, float]
        This Tuple contains the new: (new Position, none, new Force)
        """
        f = system.potential.ene
        f_prime = system.potential.force

        self.oldpos = system.current_state.position
        self.newPos = np.squeeze(fmin_cg(f=f, fprime=f_prime, x0=self.oldpos, epsilon=self.epsilon, maxiter=1, disp=False))

        move_vector = self.oldpos - self.newPos

        # Scale if move vector is bigger than maxstep
        prefactor = abs(self.maxStepSize / move_vector) if (self.oldpos - self.newPos != 0) else 0
        if not isinstance(self.maxStepSize, type(None)) and prefactor < 1:
            scaled_move_vector = prefactor * move_vector
            self.newPos = self.oldpos - scaled_move_vector

        return self.newPos, np.nan, move_vector

    def optimize(self, potential, x0, maximal_iterations=100) -> dict:
        """
        Performs the optimization on the basis of the scipy.optimize.fmin_cg function. Raises custim
        errors if the optimization was not sucessfull.

        Parameters
        ----------
        potential : potentialType
            Energy potential of the system
        x0 : array
            A user-supplied initial estimate of xopt, the optimal value of x. Usually the starting position
        maximal_iterations : int, optional
            Maximum number of iterations to perform, by default 100

        Returns
        -------
        dict
            Dictionary that contains the optimized position ("optimal_position") with its corresponding potential
            energy "minimal_potential_energy", the number of iterations needed to converge "used_iterations"
            and a list of arrays, containing the results at each iteration "position_trajectory"

        Raises
        ------
        ValueError
            Value Error is raised if the optimization did not converged or if the optimization run into an
            internal error.

        """

        cg_out = fmin_cg(
            f=potential.ene, fprime=potential.force, x0=x0, epsilon=self.epsilon, maxiter=maximal_iterations, full_output=True, retall=True
        )
        opt_position, Vmin, function_iterations, gradient_iterations, warn_flag, traj_positions = cg_out

        if warn_flag == 1:
            raise ValueError("Did not converge with the maximal number of iterations")
        elif warn_flag == 2:
            raise ValueError("Function values did not change! Error in function? precision problem?")
        elif warn_flag == 3:
            raise ValueError("encountered NaN")

        return {
            "optimal_position": opt_position,
            "minimal_potential_energy": Vmin,
            "used_iterations": function_iterations,
            "position_trajectory": traj_positions,
        }


# alternative class names
class cg(conjugate_gradient):
    pass
