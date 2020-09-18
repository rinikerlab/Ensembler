"""
Free Energy Calculations:
    This module contains functions for free energy calculations
    author: Gerhard König, Benjamin Schroeder
"""
# Generic Typing
from decimal import Decimal
from numbers import Number
from typing import Iterable

# Calculations
import numpy as np
import sympy as sp
from scipy import constants as const


class _FreeEnergyCalculator:
    constants: dict
    equation: sp.function
    simplified_equation: sp.function

    exp = np.vectorize(lambda x: x.exp())  # this function deals with decimals to
    ln = lambda self, x: x.ln()
    J_to_cal: float = 0.239005736
    k, T, Vi, Vj = sp.symbols("k T, Vi, Vj")

    def __init__(self):
        pass

    def __str__(self):
        msg = ""
        msg += self.__class__.__name__ + "\n"
        msg += "\tEquation: " + str(self.equation) + "\n"
        msg += "\tConstants:\n\t\t" + "\n\t\t".join(
            [str(key) + ":\t" + str(value) for key, value in self.constants.items()]) + "\n"
        msg += "\tsimplified Equation: " + str(self.simplified_equation) + "\n"
        return msg

    def calculate(self, Vi: (Iterable[Number], Number), Vj: (Iterable[Number], Number)) -> float:
        raise NotImplementedError("This Function needs to be Implemented")

    def _update_function(self):
        self.simplified_equation = self.equation.subs(self.constants)

    @classmethod
    def _prepare_type(self, *arrays):
        return tuple(map(lambda arr: np.array(list(map(lambda x: Decimal(x), arr))), arrays))

    @classmethod
    def get_equation(cls) -> sp.function:
        """
        get_equation returns the symoblic Equation

        :return: symbolic implementation of Zwanzig
        :rtype: sp.function
        """
        return cls.equation

    @classmethod
    def get_equation_simplified(cls) -> sp.function:
        cls._update_function()
        return cls.simplified_equation

    def set_parameters(self):
        """
        set_parameters setter for the parameter

        """
        raise NotImplementedError("This function needs to be implemented")


class zwanzigEquation(_FreeEnergyCalculator):
    """Zwanzig Equation

    This class is a nice wrapper for the zwanzig Equation.

    :raises ValueError: The input was wrongly formatted
    :raises OverflowError: The calculation overfloweds
    """
    k, T, Vi, Vj = sp.symbols("k T, Vi, Vj")
    equation: sp.function = -(1 / (k * T)) * sp.log(sp.exp(-(1 / (k * T)) * (Vj - Vi)))
    constants: dict

    def __init__(self, T: float = 298, k: float = const.k * const.Avogadro, kT: bool = False, kJ: bool = False,
                 kCal: bool = False):
        """
        __init__ Here you can set Class wide the parameters T and k for the Zwanzig Equation

        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """

        self.constants = {}
        if (kT):
            self.set_parameters(T=Decimal(1), k=Decimal(1))
        elif (kJ):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro / 1000))
        elif (kCal):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro * self.J_to_cal / 1000))
        else:
            self.set_parameters(T=Decimal(T), k=Decimal(k))

        self._update_function()

    def calculate(self, Vi: (Iterable[Number], Number), Vj: (Iterable[Number], Number)) -> float:
        """zwanzig

        Calculate a free energy difference with the Zwanzig equation (aka exponential formula or thermodynamic perturbation).
        The initial state of the free energy difference is denoted as 0, the final state is called 1.
        The function expects two arrays of size n with potential energies. The first array, u00, contains the potential energies of a set
        of Boltzmann-weighted conformations of an MD or MC trajectory of the initial state, analyzed with the Hamiltonian of the
        initial state. The second array, u01 , contains the potential energies of a trajectory of the initial state that was
        analyzed with the potential energy function of the final state. The variable kT expects the product of the Boltzmann
        constant with the temperature that was used to generate the trajectory in the respective units of the potential energies.

        See Zwanzig, R. W. J. Chem. Phys. 1954, 22, 1420-1426. doi:10.1063/1.1740409

        Parameters
        ----------
        Vi : np.array
        Vj : np.array

        Returns
        -------
        float
            free energy difference

        """
        return float(self._calculate_efficient(Vi, Vj))

    def _calculate_implementation_bruteForce(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """

        Parameters
        ----------
        Vi : np.array
        Vj : np.array
        Vr : np.array

        Returns
        -------
        float
            free energy difference

        """
        if (not (len(Vi) == len(Vj))):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi) + " \t " + str(len(Vj))) + "\n")

        # Typcasting
        Vi, Vj = self._prepare_type(Vi, Vj)

        beta = 1 / (self.constants[self.k] * self.constants[self.T])  # calculate Beta

        # Calculate the potential energy difference in reduced units of kT
        dVij = - beta * (Vj - Vi)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij = self.exp(dVij)
        except OverflowError:
            raise OverflowError(
                "Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # average of exponents
        meandVij = np.mean(edVij)

        # Return free energy difference
        try:
            # dF = zwanzigEquation.calculate(Vi, Vr) - zwanzigEquation.calculate(Vj, Vr)
            dF = - (1 / beta) * (meandVij).ln()
        except ValueError:
            raise ValueError(
                "Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference ")

        return dF

    def _calculate_meanEfficient(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """zwanzig

        Calculate a free energy difference with the Zwanzig equation (aka exponential formula or thermodynamic perturbation).
        The initial state of the free energy difference is denoted as 0, the final state is called 1.
        The function expects two arrays of size n with potential energies. The first array, u00, contains the potential energies of a set
        of Boltzmann-weighted conformations of an MD or MC trajectory of the initial state, analyzed with the Hamiltonian of the
        initial state. The second array, u01 , contains the potential energies of a trajectory of the initial state that was
        analyzed with the potential energy function of the final state. The variable kT expects the product of the Boltzmann
        constant with the temperature that was used to generate the trajectory in the respective units of the potential energies.

        This is an efficient more overflow robust implementation of the Zwanzig Equation.

        @Author: Gerhard König
        See Zwanzig, R. W. J. Chem. Phys. 1954, 22, 1420-1426. doi:10.1063/1.1740409

        Parameters
        ----------
        Vi : np.array
        Vj : np.array

        Returns
        -------
        float
            free energy difference

        """
        if (not (len(Vi) == len(Vj))):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi) + " \t " + str(len(Vj))) + "\n")

        Vi, Vj = self._prepare_type(Vi, Vj)
        beta = 1 / (self.constants[self.k] * self.constants[self.T])

        # Calculate the potential energy difference in reduced units of kT
        dVij = - beta * np.subtract(Vj, Vi)

        # Calculate offset for increased numerical stability
        meandVij = np.mean(dVij)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij = self.exp(dVij - meandVij)
        except OverflowError:
            raise OverflowError(
                "Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # Return free energy difference
        try:
            dF = - (1 / beta) * ((np.mean(edVij)).ln() + meandVij)
        except ValueError:
            raise ValueError(
                "Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference " + str(
                    np.mean(edVij)))

        return dF

    def _calculate_efficient(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """zwanzig

        Calculate a free energy difference with the Zwanzig equation (aka exponential formula or thermodynamic perturbation).
        The initial state of the free energy difference is denoted as 0, the final state is called 1.
        The function expects two arrays of size n with potential energies. The first array, u00, contains the potential energies of a set
        of Boltzmann-weighted conformations of an MD or MC trajectory of the initial state, analyzed with the Hamiltonian of the
        initial state. The second array, u01 , contains the potential energies of a trajectory of the initial state that was
        analyzed with the potential energy function of the final state. The variable kT expects the product of the Boltzmann
        constant with the temperature that was used to generate the trajectory in the respective units of the potential energies.

        This is an efficient more overflow robust implementation of the Zwanzig Equation.

        @Author: Gerhard König
        See Zwanzig, R. W. J. Chem. Phys. 1954, 22, 1420-1426. doi:10.1063/1.1740409

        Parameters
        ----------
        Vi : np.array
        Vj : np.array

        Returns
        -------
        float
            free energy difference

        """
        if (not (len(Vi) == len(Vj))):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi) + " \t " + str(len(Vj))) + "\n")

        Vi, Vj = self._prepare_type(Vi, Vj)
        beta = 1 / (self.constants[self.k] * self.constants[self.T])

        # Calculate the potential energy difference in reduced units of kT
        dVij = - beta * np.subtract(Vj, Vi)

        # Return free energy difference
        from scipy import special as s
        dF = - np.float(1 / beta) * s.logsumexp(np.array(dVij, dtype=np.float), b=1/len(dVij))
        return dF

    def set_parameters(self, T: float = None, k: float = None):
        """
        set_parameters setter for the parameter

        :param T: Temperature in Kelvin, defaults to None
        :type T: float, optional
        :param k: Boltzmann Constant, defaults to None
        :type k: float, optional
        """
        if (isinstance(T, (Number, Decimal))):
            self.constants.update({self.T: Decimal(T)})

        if (isinstance(k, (Number, Decimal))):
            self.constants.update({self.k: Decimal(k)})
        self._update_function()


class zwanzig(zwanzigEquation):
    pass


class threeStateZwanzigReweighting(zwanzigEquation):
    k, T, Vi, Vj, Vr = sp.symbols("k T Vi Vj Vr")
    equation: sp.function = -(1 / (k * T)) * (
                sp.log(sp.exp(-(1 / (k * T)) * (Vi - Vr))) - sp.log(sp.exp(-(1 / (k * T)) * (Vj - Vr))))

    def __init__(self, kCal: bool = False, T: float = 298, k: float = const.k * const.Avogadro, kT: bool = False,
                 kJ: bool = False):
        """
        __init__ Here you can set Class wide the parameters T and k for the Zwanzig Equation

        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """
        super().__init__(kCal=kCal, T=T, k=k, kT=kT, kJ=kJ)

    def calculate(self, Vi: (Iterable[Number], Number), Vj: (Iterable[Number], Number),
                  Vr: (Iterable[Number], Number)) -> float:
        """reweighted Zwanzig

        Parameters
        ----------
        Vi : np.array
        Vj : np.array
        Vr : np.array

        Returns
        -------
        float
            free energy difference

        """
        return float(self._calculate_implementation_useZwanzig(Vi=Vi, Vj=Vj, Vr=Vr))

    def _calculate_implementation_useZwanzig(self, Vi: (Iterable, Number), Vj: (Iterable, Number),
                                             Vr: (Iterable[Number], Number)) -> float:
        """

        Parameters
        ----------
        Vi : np.array
        Vj : np.array
        Vr : np.array

        Returns
        -------
        float
            free energy difference

        """
        if (not (len(Vi) == len(Vj) == len(Vr))):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi) + " \t " + str(len(Vj))) + "\n")

        # Type Casting
        Vi, Vj, Vr = self._prepare_type(Vi, Vj, Vr)

        # Build Zwanzig
        zwanz = zwanzigEquation()
        zwanz.constants = self.constants

        # Calc
        #   VR <- V1
        dF1r = zwanz.calculate(Vi=Vi, Vj=Vr)
        #   VR <- V2
        dF2r = zwanz.calculate(Vi=Vj, Vj=Vr)
        #   V1 <- VR - VR -> V2
        dF = dF1r - dF2r

        return dF


class dfEDS(threeStateZwanzigReweighting):
    pass


class bennetAcceptanceRatio(_FreeEnergyCalculator):
    k, T, beta, C, Vi_i, Vj_i, Vi_j, Vj_j = sp.symbols("k T beta C  Vi_i Vj_i Vi_j Vj_j")
    equation: sp.function = (1 / (k * T)) * (
                sp.log(sp.exp((1 / (k * T)) * (Vi_j - Vj_j + C))) - sp.log(sp.exp((1 / (k * T)) * (Vj_i - Vi_i + C))))
    constants: dict = {T: 298, k: const.k * const.Avogadro, C: Decimal(0)}

    # Numeric parameters
    convergence_radius: float
    max_iterations: int
    min_iterations: int

    def __init__(self, C: float = 0.0, T: float = 298, k: float = const.k * const.Avogadro,
                 kT: bool = False, kJ: bool = False, kCal: bool = False,
                 convergence_radius: float = 10 ** (-5), max_iterations: int = 100, min_iterations: int = 1):
        """
        __init__ Here you can set Class wide the parameters T and k for the bennet acceptance ration (BAR) Equation

        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """

        self.constants = {}
        if (kT):
            self.set_parameters(T=Decimal(1), k=Decimal(1), C=C)
        elif (kJ):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro / 1000), C=C)
        elif (kCal):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro * self.J_to_cal / 1000), C=C)
        else:
            self.set_parameters(T=Decimal(T), k=Decimal(k))

        # deal with numeric params:
        self.convergence_radius = convergence_radius
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        self._update_function()

    def calculate(self, Vi_i: (Iterable[Number], Number), Vj_i: (Iterable[Number], Number),
                  Vi_j: (Iterable[Number], Number), Vj_j: (Iterable[Number], Number)) -> float:
        """reweighted Zwanzig

        Parameters
        ----------
        Vi_i : np.array
        Vj_i : np.array
        Vi_j : np.array
        Vj_j : np.array

        Returns
        -------
        float
            free energy difference

        """
        return self._calculate_optimize(Vi_i, Vj_i, Vi_j, Vj_j)

    def _calc_bar(self, C: Decimal, Vj_i: np.array, Vi_i: np.array, Vi_j: np.array, Vj_j: np.array) -> Decimal:

        # Calculate the potential energy difference in reduced units of kT
        dV_i = self.constants[self.beta] * ((Vj_i - Vi_i) - C)
        dV_j = self.constants[self.beta] * ((Vi_j - Vj_j) + C)

        # Exponentiate to obtain fermi(-Delta U/kT)
        try:
            ferm_dV_i = 1 / (self.exp(dV_i) + 1)
            ferm_dV_j = 1 / (self.exp(dV_j) + 1)
        except OverflowError:
            raise OverflowError(
                "Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # get average
        mean_edV_i = np.mean(ferm_dV_i)
        mean_edV_j = np.mean(ferm_dV_j)

        # Return free energy difference
        try:
            ddF = -(1 / self.constants[self.beta]) * self.ln(mean_edV_j / mean_edV_i)
        except ValueError as err:
            raise ValueError(
                "BAR Error: Problems taking logarithm of the average exponentiated potential energy difference " + str(
                    err.args))

        dF = ddF - C
        return dF

    def _calculate_optimize(self, Vi_i: (Iterable[Number], Number), Vj_i: (Iterable[Number], Number),
                            Vi_j: (Iterable[Number], Number), Vj_j: (Iterable[Number], Number), C0: float = 0,
                            verbose: bool = True) -> float:
        """
            method  bisection

        Parameters
        ----------
        Vi_i : np.array
        Vj_i : np.array
        Vi_j : np.array
        Vj_j : np.array

        Returns
        -------
        float
            free energy difference

        """
        if (
        not ((len(Vi_i) == len(Vi_i)) and (len(Vi_j) == len(Vj_j)))):  # I and J simulation don't need the same length.
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi_i) + " \t " + str(len(Vj_i))) + "\n" + str(len(Vi_j) + " \t " + str(len(Vj_j))) + "\n")

        # Type Cast
        Vi_i, Vj_i, Vi_j, Vj_j = self._prepare_type(Vj_i, Vi_i, Vi_j, Vj_j)

        # Calc Beta
        self.constants.update({self.beta: Decimal(1 / (self.constants[self.k] * self.constants[self.T]))})
        # given C?
        if (not isinstance(C0, type(None))):
            self.constants.update({self.C: C0})

        # optimization scheme:
        if (verbose): print("Iterate: \tconvergence raidus: " + str(self.convergence_radius))
        iteration = 0
        convergence = self.convergence_radius + 1
        while self.max_iterations > iteration:

            dF = self._calc_bar(C=self.constants[self.C], Vj_i=Vj_i, Vi_i=Vi_i, Vi_j=Vi_j, Vj_j=Vj_j)  # calc dF

            newC = dF
            convergence = abs(self.constants[self.C] - dF)

            if (verbose): print("Iteration: " + str(iteration) + "\tdF: " + str(dF) , "\tconvergence", convergence)

            if (convergence > self.convergence_radius or  self.min_iterations > iteration):
                iteration += 1
                self.constants.update({self.C: newC})
            else:
                break
        print()
        if (iteration >= self.max_iterations):
            raise Exception(
                "BAR is not converged after " + str(iteration) + " steps. stopped at: " + str(self.constants[self.C]))
        print("Final Iterations: ", iteration, " Result: ", dF)

        return float(dF)

    def set_parameters(self, C: float = None, T: float = None, k: float = None):
        """
        set_parameters setter for the parameter

        :param T: Temperature in Kelvin, defaults to None
        :type T: float, optional
        :param k: Boltzmann Constant, defaults to None
        :type k: float, optional
        """
        if (isinstance(T, Number)):
            self.constants.update({self.T: Decimal(T)})

        if (isinstance(k, Number)):
            self.constants.update({self.k: Decimal(k)})

        if (isinstance(C, Number)):
            self.constants.update({self.C: Decimal(C)})

        self._update_function()

# alternative class names
class bar(bennetAcceptanceRatio):
    pass


"""
class multistatebennetAcceptanceRatio(_FreeEnergyCalculator):
    k, T, beta, C, Vi_i, Vj_i, Vi_j, Vj_j = sp.symbols("k T beta C  Vi_i Vj_i Vi_j Vj_j")
    equation: sp.function = (1 / (k * T)) * (
                sp.log(sp.exp((1 / (k * T)) * (Vi_j - Vj_j + C))) - sp.log(sp.exp((1 / (k * T)) * (Vj_i - Vi_i + C))))
    constants: dict = {T: 298, k: const.k * const.Avogadro, C: Decimal(0)}

    # Numeric parameters
    convergence_radius: float
    max_iterations: int
    min_iterations: int

    def __init__(self, C: float = 0.0, T: float = 298, k: float = const.k * const.Avogadro,
                 kT: bool = False, kJ: bool = False, kCal: bool = False,
                 convergence_radius: float = 10 ** (-5), max_iterations: int = 100, min_iterations: int = 1):
        \"""
        __init__ Here you can set Class wide the parameters T and k for the bennet acceptance ration (BAR) Equation

        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        \"""

        self.constants = {}
        if (kT):
            self.set_parameters(T=Decimal(1), k=Decimal(1), C=C)
        elif (kJ):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro / 1000), C=C)
        elif (kCal):
            self.set_parameters(T=Decimal(T), k=Decimal(const.k * const.Avogadro * self.J_to_cal / 1000), C=C)
        else:
            self.set_parameters(T=Decimal(T), k=Decimal(k))

        # deal with numeric params:
        self.convergence_radius = convergence_radius
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        self._update_function()

    def calculate(self, Vi_i: (Iterable[Number], Number), Vj_i: (Iterable[Number], Number),
                  Vi_j: (Iterable[Number], Number), Vj_j: (Iterable[Number], Number)) -> float:
        \"""reweighted Zwanzig

        Parameters
        ----------
        Vi_i : np.array
        Vj_i : np.array
        Vi_j : np.array
        Vj_j : np.array

        Returns
        -------
        float
            free energy difference

        \"""
        return self._calculate_optimize(Vi_i, Vj_i, Vi_j, Vj_j)

    def _calc_mbar(self, C: Decimal, Vj_i: np.array, Vi_i: np.array, Vi_j: np.array, Vj_j: np.array) -> Decimal:

        #state sampled k
        downfrac = nstatesk * np.sum([np.exp(-self.constants[self.beta] * (C - V_jk)) for V_jk in V_xks])
        topfracs = []
        A_iks = []
        for V_ik in V_xks:
            topfrac = np.exp(-self.constants[self.beta]*V_ik)
            print("tops", topfrac)
            topfracs.append(topfrac)
            A_ik =topfrac/downfrac

        dF = ddF - C
        return dF

    def _calculate_optimize(self, Vi_i: (Iterable[Number], Number), Vj_i: (Iterable[Number], Number),
                            Vi_j: (Iterable[Number], Number), Vj_j: (Iterable[Number], Number), C0: float = 0,
                            verbose: bool = True) -> float:
        \"""
            method  bisection

        Parameters
        ----------
        Vi_i : np.array
        Vj_i : np.array
        Vi_j : np.array
        Vj_j : np.array

        Returns
        -------
        float
            free energy difference

        \"""
        if (
        not ((len(Vi_i) == len(Vi_i)) and (len(Vi_j) == len(Vj_j)))):  # I and J simulation don't need the same length.
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: " + str(
                    len(Vi_i) + " \t " + str(len(Vj_i))) + "\n" + str(len(Vi_j) + " \t " + str(len(Vj_j))) + "\n")

        # Type Cast
        Vi_i, Vj_i, Vi_j, Vj_j = self._prepare_type(Vj_i, Vi_i, Vi_j, Vj_j)

        # Calc Beta
        self.constants.update({self.beta: Decimal(1 / (self.constants[self.k] * self.constants[self.T]))})
        # given C?
        if (not isinstance(C0, type(None))):
            self.constants.update({self.C: C0})

        # optimization scheme:
        if (verbose): print("Iterate: \tconvergence raidus: " + str(self.convergence_radius))
        iteration = 0
        convergence = self.convergence_radius + 1
        while self.max_iterations > iteration:

            dF = self._calc_bar(C=self.constants[self.C], Vj_i=Vj_i, Vi_i=Vi_i, Vi_j=Vi_j, Vj_j=Vj_j)  # calc dF

            newC = dF
            convergence = abs(self.constants[self.C] - dF)

            if (verbose): print("Iteration: " + str(iteration) + "\tdF: " + str(dF) , "\tconvergence", convergence)

            if (convergence > self.convergence_radius or  self.min_iterations > iteration):
                iteration += 1
                self.constants.update({self.C: newC})
            else:
                break
        print()
        if (iteration >= self.max_iterations):
            raise Exception(
                "BAR is not converged after " + str(iteration) + " steps. stopped at: " + str(self.constants[self.C]))
        print("Final Iterations: ", iteration, " Result: ", dF)

        return float(dF)

    def set_parameters(self, C: float = None, T: float = None, k: float = None):
        \"""
        set_parameters setter for the parameter

        :param T: Temperature in Kelvin, defaults to None
        :type T: float, optional
        :param k: Boltzmann Constant, defaults to None
        :type k: float, optional
        \"""
        if (isinstance(T, Number)):
            self.constants.update({self.T: Decimal(T)})

        if (isinstance(k, Number)):
            self.constants.update({self.k: Decimal(k)})

        if (isinstance(C, Number)):
            self.constants.update({self.C: Decimal(C)})

        self._update_function()
"""