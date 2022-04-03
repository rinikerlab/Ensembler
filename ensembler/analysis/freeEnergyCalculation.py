"""
Free Energy Calculations:
    This module contains functions for free energy calculations
    author: Gerhard König, Benjamin Schroeder
"""
# Generic Typing
from numbers import Number
from typing import Iterable

# Calculations
import numpy as np
import sympy as sp
import mpmath as mp
from scipy import constants as const


class _FreeEnergyCalculator:
    constants: dict
    equation: sp.Function
    simplified_equation: sp.Function

    exp = np.vectorize(lambda x: mp.exp(x))  # this function deals with decimals to
    ln = lambda self, x: mp.ln(x)
    fermifunc = np.vectorize(lambda x, beta: 1 / (1 + mp.exp(beta * x)))

    J_to_cal: float = 0.239005736
    k, T, Vi, Vj = sp.symbols("k T, Vi, Vj")

    def __init__(self):
        pass

    def __str__(self):
        msg = ""
        msg += self.__class__.__name__ + "\n"
        msg += "\tEquation: " + str(self.equation) + "\n"
        msg += "\tConstants:\n\t\t" + "\n\t\t".join([str(key) + ":\t" + str(value) for key, value in self.constants.items()]) + "\n"
        msg += "\tsimplified Equation: " + str(self.simplified_equation) + "\n"
        return msg

    def calculate(self, Vi: (Iterable[Number], Number), Vj: (Iterable[Number], Number)) -> float:
        raise NotImplementedError("This Function needs to be Implemented")

    def _update_function(self):
        self.simplified_equation = self.equation.subs(self.constants)

    @classmethod
    def _prepare_type(self, *arrays):
        return tuple(map(lambda arr: np.array(list(map(lambda x: np.float(x), arr)), ndmin=1), arrays))

    @classmethod
    def get_equation(cls) -> sp.Function:
        """
        get_equation returns the symoblic Equation

        :return: symbolic implementation of Zwanzig
        :rtype: sp.Function
        """
        return cls.equation

    @classmethod
    def get_equation_simplified(cls) -> sp.Function:
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

    dF = - \beta \ln(\langle e^(-beta(V_j-V_i)) \rangle)

    """

    k, T, Vi, Vj = sp.symbols("k T, Vi, Vj")
    equation: sp.Function = -(1 / (k * T)) * sp.log(sp.exp(-(1 / (k * T)) * (Vj - Vi)))
    constants: dict

    def __init__(self, T: float = 298, k: float = const.k * const.Avogadro, kT: bool = False, kJ: bool = False, kCal: bool = False):
        """
        __init__
            Here you can set Class wide the parameters T and k for the Zwanzig Equation

        Parameters
        ----------
        T: float, optional
            Temperature in Kelvin, defaults to 398
        k: float, optional
            boltzmann Constant, defaults to const.k*const.Avogadro
        kT: bool, optional
            overwrites T and k to set all results in units of $k_bT$
        kJ: bool, optional
            overwrites k to get the Boltzman constant with units kJ/(mol*K)
        kCal: bool, optional
            overwrites k to get the Boltzman constant with units kcal/(mol*K)

        """

        self.constants = {}
        if kT:
            self.set_parameters(T=1, k=1)
        elif kJ:
            self.set_parameters(T=T, k=const.k * const.Avogadro / 1000)
        elif kCal:
            self.set_parameters(T=T, k=const.k * const.Avogadro * self.J_to_cal / 1000)
        else:
            self.set_parameters(T=T, k=k)

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
            Potential energies of state I
        Vj : np.array
            Potential energies of state J

        Returns
        -------
        float
            free energy difference

        """
        return float(self._calculate_mpmath(Vi=Vi, Vj=Vj))

    def _calculate_implementation_bruteForce(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """
            _calculate_implementation_bruteForce
                This is a plain implementation of the zwanzig equation. It is not very numerical robust
        Parameters
        ----------
        Vi
        Vj

        Parameters
        ----------
        Vi : np.array
            Potential energies of state I
        Vj : np.array
            Potential energies of state J

        Returns
        -------
        float
            free energy difference
        """

        if not (len(Vi) == len(Vj)):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "
                + str(len(Vi) + " \t " + str(len(Vj)))
                + "\n"
            )

        # Typcasting
        Vi, Vj = self._prepare_type(Vi, Vj)

        beta = 1 / (self.constants[self.k] * self.constants[self.T])  # calculate Beta

        # Calculate the potential energy difference in reduced units of kT
        dVij = -beta * (Vj - Vi)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij = self.exp(dVij)
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # average of exponents
        meandVij = np.mean(edVij)

        # Return free energy difference
        try:
            # dF = zwanzigEquation.calculate(Vi, Vr) - zwanzigEquation.calculate(Vj, Vr)
            dF = -(1 / beta) * (meandVij).ln()
        except ValueError:
            raise ValueError("Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference ")

        return dF

    def _calculate_meanEfficient(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """
        _calculate_meanEfficient

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

            _calculate_implementation_bruteForce
                This is a plain implementation of the zwanzig equation. It is not very numerical robust

        Parameters
        ----------
        Vi : np.array
            Potential energies of state I
        Vj : np.array
            Potential energies of state J

        Returns
        -------
        float
            free energy difference

        """
        if not (len(Vi) == len(Vj)):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "
                + str(len(Vi) + " \t " + str(len(Vj)))
                + "\n"
            )

        Vi, Vj = self._prepare_type(Vi, Vj)
        beta = 1 / (self.constants[self.k] * self.constants[self.T])

        # Calculate the potential energy difference in reduced units of kT
        dVij = -beta * (Vj - Vi)

        # Calculate offset for increased numerical stability
        meandVij = np.mean(dVij)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij = self.exp(dVij - meandVij)
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # Return free energy difference
        try:
            dF = -(1 / beta) * ((np.mean(edVij)).ln() + meandVij)
        except ValueError:
            raise ValueError(
                "Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference " + str(np.mean(edVij))
            )

        return dF

    def _calculate_efficient(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """
        _calculate_efficient

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

        $dF = \frac{1}{\beta} * \ln(\langlee^{-\beta * (V_j-V_i)}\rangle)$

        Parameters
        ----------
        Vi : np.array
            Potential energies of state I
        Vj : np.array
            Potential energies of state J

        Returns
        -------
        float
            free energy difference

        """
        if not (len(Vi) == len(Vj)):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "
                + str(len(Vi) + " \t " + str(len(Vj)))
                + "\n"
            )

        Vi, Vj = self._prepare_type(Vi, Vj)
        beta = 1 / (self.constants[self.k] * self.constants[self.T])

        # Calculate the potential energy difference in reduced units of kT
        dVij = -beta * (Vj - Vi)

        # Return free energy difference
        from scipy import special as s

        dF = -np.float(1 / beta) * s.logsumexp(np.array(dVij, dtype=np.float), b=1 / len(dVij))
        return dF

    def _calculate_mpmath(self, Vi: (Iterable, Number), Vj: (Iterable, Number)) -> float:
        """
        implementation of zwanzig with mpmath package, another way of having a robust variant,
        but this one is very close to the initial equation thanks to the mpmath package.

        $dF = \frac{1}{\beta} * \ln(\langlee^{-\beta * (V_j-V_i)}\rangle)$

        Parameters
        ----------
        Vi : np.array
            Potential energies of state I
        Vj : np.array
            Potential energies of state J

        Returns
        -------
        float
            free energy difference

        """
        beta = np.float(1 / (self.constants[self.k] * self.constants[self.T]))

        return -(1 / beta) * np.float(mp.ln(np.mean(list(map(mp.exp, -beta * (np.array(Vj, ndmin=1) - np.array(Vi, ndmin=1)))))))

    def set_parameters(self, T: float = None, k: float = None):
        """
            set_parameters setter for the parameters T and k

        Parameters
        ----------
        T: float, optional
            Temperature in Kelvin, defaults to 398
        k: float, optional
            boltzmann Constant, defaults to const.k*const.Avogadro

        """

        if isinstance(T, (Number)):
            self.constants.update({self.T: T})

        if isinstance(k, (Number)):
            self.constants.update({self.k: k})
        self._update_function()


class zwanzig(zwanzigEquation):
    pass


class threeStateZwanzig(zwanzigEquation):
    """
    this class provides the implementation for the Free energy calculation with EDS.
    It calculates the free energy via the reference state.

    $dF = dF_{BR}-dF_{AR} = \frac{1}{\beta} * ( \ln(\langle e^{-\beta * (V_j-V_R)}\rangle) - \ln(\langle e^{-\beta * (V_i-V_R)}\rangle))$
    """

    k, T, Vi, Vj, Vr = sp.symbols("k T Vi Vj Vr")
    equation: sp.Function = -(1 / (k * T)) * (sp.log(sp.exp(-(1 / (k * T)) * (Vi - Vr))) - sp.log(sp.exp(-(1 / (k * T)) * (Vj - Vr))))

    def __init__(self, kCal: bool = False, T: float = 298, k: float = const.k * const.Avogadro, kT: bool = False, kJ: bool = False):
        """
        __init__
            this class provides the implementation for the Free energy calculation with EDS.
            It calculates the free energy via the reference state.

        Parameters
        ----------
        T: float, optional
            Temperature in Kelvin, defaults to 398
        k: float, optional
            boltzmann Constant, defaults to const.k*const.Avogadro
        kT: bool, optional
            overwrites T and k to set all results in units of $k_bT$
        kJ: bool, optional
            overwrites k to get the Boltzman constant with units kJ/(mol*K)
        kCal: bool, optional
            overwrites k to get the Boltzman constant with units kcal/(mol*K)
        """
        super().__init__(kCal=kCal, T=T, k=k, kT=kT, kJ=kJ)

    def calculate(self, Vi: (Iterable[Number], Number), Vj: (Iterable[Number], Number), Vr: (Iterable[Number], Number)) -> float:
        """
            calculate
                this method calculates the zwanzig equation via the intermediate reference state R using the Zwanzig equation.

        Parameters
        ----------
        Vi : np.array
            the potential energy of stateI while sampling stateR
        Vj : np.array
            the potential energy of stateJ while sampling stateR
        Vr : np.array
            the potential energy of stateR while sampling stateR

        Returns
        -------
        float
            free energy difference

        """
        return float(self._calculate_implementation_useZwanzig(Vi=Vi, Vj=Vj, Vr=Vr))

    def _calculate_implementation_useZwanzig(self, Vi: (Iterable, Number), Vj: (Iterable, Number), Vr: (Iterable[Number], Number)) -> float:
        """
            calculate
                this method calculates the zwanzig equation via the intermediate reference state R using the Zwanzig equation.
                it directly accesses the zwanzig implmentation.

        Parameters
        ----------
        Vi : np.array
            the potential energy of stateI while sampling stateR
        Vj : np.array
            the potential energy of stateJ while sampling stateR
        Vr : np.array
            the potential energy of stateR while sampling stateR

        Returns
        -------
        float
            free energy difference

        """
        if not (len(Vi) == len(Vj) == len(Vr)):
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "
                + str(len(Vi) + " \t " + str(len(Vj)))
                + "\n"
            )

        # Type Casting
        Vi, Vj, Vr = self._prepare_type(Vi, Vj, Vr)

        # Build Zwanzig
        zwanz = zwanzigEquation()
        zwanz.constants = self.constants

        # Calc
        #   V1 <- VR == V1-Vr
        dF1r = zwanz.calculate(Vi=Vr, Vj=Vi)
        #   V2 <- VR == V2-Vr
        dF2r = zwanz.calculate(Vi=Vr, Vj=Vj)
        #   (V1 <- VR) - (VR -> V2)
        dF = dF2r - dF1r

        return dF


class dfEDS(threeStateZwanzig):
    pass


class bennetAcceptanceRatio(_FreeEnergyCalculator):
    """
    This class implements the BAR method.
    $dF = -\frac{1}{\beta} * \ln( \frac{f\langle(V_j-V_i+C)\rangle_i}{f\langle(V_i-V_j-C)\rangle_j})+C$
    with :
    $ f(x) = \frac{1}{1+e^(\beta x)}$ - fermi function

    """

    k, T, beta, C, Vi_i, Vj_i, Vi_j, Vj_j = sp.symbols("k T beta C  Vi_i Vj_i Vi_j Vj_j")
    equation: sp.Function = (1 / (k * T)) * (
        sp.log(sp.exp((1 / (k * T)) * (Vi_j - Vj_j + C))) - sp.log(sp.exp((1 / (k * T)) * (Vj_i - Vi_i + C)))
    )
    constants: dict = {T: 298, k: const.k * const.Avogadro, C: Number}

    # Numeric parameters
    convergence_radius: float
    max_iterations: int
    min_iterations: int

    def __init__(
        self,
        C: float = 0.0,
        T: float = 298,
        k: float = const.k * const.Avogadro,
        kT: bool = False,
        kJ: bool = False,
        kCal: bool = False,
        convergence_radius: float = 10 ** (-5),
        max_iterations: int = 500,
        min_iterations: int = 1,
    ):
        """
        __init__
        Here you can set Class wide the parameters T and k for the bennet acceptance ration (BAR) Equation

        Parameters
        ----------
        C: float, optional
            is the initial guess of the free Energy.
        T: float, optional
            Temperature in Kelvin, defaults to 398
        k: float, optional
            boltzmann Constant, defaults to const.k*const.Avogadro
        kT: bool, optional
            overwrites T and k to set all results in units of $k_bT$
        kJ: bool, optional
            overwrites k to get the Boltzman constant with units kJ/(mol*K)
        kCal: bool, optional
            overwrites k to get the Boltzman constant with units kcal/(mol*K)
        convergence_radius: float, optional
            when is the result converged? if the deviation of one to another iteration is below the convergence radius.
        max_iterations: int, optional
            maximal number of iterations.
        min_iterations: int, optional
            minimal number of iterations
        """

        self.constants = {}
        if kT:
            self.set_parameters(T=1, k=1, C=C)
        elif kJ:
            self.set_parameters(T=T, k=const.k * const.Avogadro / 1000, C=C)
        elif kCal:
            self.set_parameters(T=T, k=const.k * const.Avogadro * self.J_to_cal / 1000, C=C)
        else:
            self.set_parameters(T=T, k=k)

        # deal with numeric params:
        self.convergence_radius = convergence_radius
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        self._update_function()

    def calculate(
        self, Vi_i: Iterable[Number], Vj_i: Iterable[Number], Vi_j: Iterable[Number], Vj_j: Iterable[Number], verbose: bool = False
    ) -> float:
        """
            calculate
                this function is calculating the free energy difference of two states with the BAR method.


        Parameters
        ----------
        Vi_i : np.array
            potential energies of stateI while sampling stateI
        Vj_i : np.array
             potential energies of stateJ while sampling stateI
        Vi_j : np.array
             potential energies of stateI while sampling stateJ
        Vj_j : np.array
             potential energies of stateJ while sampling stateJ

        Returns
        -------
        float
            free energy difference

        """
        return self._calculate_optimize(Vi_i, Vj_i, Vi_j, Vj_j, verbose=verbose)

    def _calc_bar(self, C: Number, Vj_i: np.array, Vi_i: np.array, Vi_j: np.array, Vj_j: np.array) -> Number:
        """
            _calc_bar
                this function is calculating the free energy difference of two states for one iteration of the BAR method.
                 It is implemented straight forwad, but therefore not very numerical stable.


        Parameters
        ----------
        Vi_i : np.array
            potential energies of stateI while sampling stateI
        Vj_i : np.array
             potential energies of stateJ while sampling stateI
        Vi_j : np.array
             potential energies of stateI while sampling stateJ
        Vj_j : np.array
             potential energies of stateJ while sampling stateJ

        Returns
        -------
        float
            free energy difference

        """
        # Calculate the potential energy difference in reduced units of kT
        dV_j = (Vi_j - Vj_j) + C
        dV_i = (Vj_i - Vi_i) - C

        # Exponentiate to obtain fermi(-Delta U/kT)
        try:
            ferm_dV_i = 1 / (1 + self.exp(self.constants[self.beta] * dV_i))
            ferm_dV_j = 1 / (1 + self.exp(self.constants[self.beta] * dV_j))
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # get average
        mean_edV_i = np.mean(ferm_dV_i)
        mean_edV_j = np.mean(ferm_dV_j)

        # Return free energy difference
        try:
            ddF = (1 / self.constants[self.beta]) * self.ln(mean_edV_j / mean_edV_i)
        except ValueError as err:
            raise ValueError(
                "BAR Error: Problems taking logarithm of the average exponentiated potential energy difference " + str(err.args)
            )

        dF = ddF + C
        return dF

    def _calc_bar_mpmath(self, C: Number, Vj_i: np.array, Vi_i: np.array, Vi_j: np.array, Vj_j: np.array) -> Number:
        """
            _calc_bar
                this function is calculating the free energy difference of two states for one iteration of the BAR method.
                 It is implemented straight forwad, but therefore not very numerical stable.


        Parameters
        ----------
        Vi_i : np.array
            potential energies of stateI while sampling stateI
        Vj_i : np.array
             potential energies of stateJ while sampling stateI
        Vi_j : np.array
             potential energies of stateI while sampling stateJ
        Vj_j : np.array
             potential energies of stateJ while sampling stateJ

        Returns
        -------
        float
            free energy difference

        """
        # Calculate the potential energy difference in reduced units of kT
        dV_j = (Vi_j - Vj_j) + C
        dV_i = (Vj_i - Vi_i) - C

        # Exponentiate to obtain fermi(-Delta U/kT)

        try:
            ferm_dV_j = self.fermifunc(dV_j, self.constants[self.beta])
            ferm_dV_i = self.fermifunc(dV_i, self.constants[self.beta])
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # get average
        mean_edV_j = np.mean(ferm_dV_j)
        mean_edV_i = np.mean(ferm_dV_i)

        # Return free energy difference
        try:
            ddF = (1 / self.constants[self.beta]) * mp.ln(mean_edV_j / mean_edV_i)
        except ValueError as err:
            raise ValueError(
                "BAR Error: Problems taking logarithm of the average exponentiated potential energy difference " + str(err.args)
            )

        return np.float(ddF + C)

    def _calculate_optimize(
        self,
        Vi_i: (Iterable[Number], Number),
        Vj_i: (Iterable[Number], Number),
        Vi_j: (Iterable[Number], Number),
        Vj_j: (Iterable[Number], Number),
        C0: float = 0,
        verbose: bool = False,
    ) -> float:
        """
        _calculate_optimize
            this function is calculating the free energy difference of two states with the BAR method.
            it iterates over the _calc_bar method and determines the convergence and the result.


        Parameters
        ----------
        Vi_i : np.array
            potential energies of stateI while sampling stateI
        Vj_i : np.array
             potential energies of stateJ while sampling stateI
        Vi_j : np.array
             potential energies of stateI while sampling stateJ
        Vj_j : np.array
             potential energies of stateJ while sampling stateJ

        Returns
        -------
        float
            free energy difference

        """
        if not ((len(Vi_i) == len(Vi_i)) and (len(Vi_j) == len(Vj_j))):  # I and J simulation don't need the same length.
            raise ValueError(
                "Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "
                + str(len(Vi_i) + " \t " + str(len(Vj_i)))
                + "\n"
                + str(len(Vi_j) + " \t " + str(len(Vj_j)))
                + "\n"
            )

        # Calc Beta
        self.constants.update({self.beta: 1 / (self.constants[self.k] * self.constants[self.T])})
        # given C?
        if not isinstance(C0, type(None)):
            self.constants.update({self.C: C0})

        # optimization scheme:
        if verbose:
            print("Iterate: \tconvergence raidus: " + str(self.convergence_radius))
        iteration = 0
        convergence = self.convergence_radius + 1
        while self.max_iterations > iteration:

            dF = self._calc_bar_mpmath(C=self.constants[self.C], Vj_i=Vj_i, Vi_i=Vi_i, Vi_j=Vi_j, Vj_j=Vj_j)  # calc dF

            newC = dF
            convergence = abs(self.constants[self.C] - dF)

            if verbose:
                print("Iteration: " + str(iteration) + "\tdF: " + str(dF), "\tconvergence", convergence)

            if convergence > self.convergence_radius or self.min_iterations > iteration:
                iteration += 1
                self.constants.update({self.C: newC})
            else:
                break
        print()
        if iteration >= self.max_iterations:
            raise Exception("BAR is not converged after " + str(iteration) + " steps. stopped at: " + str(self.constants[self.C]))
        print("Final Iterations: ", iteration, " Result: ", dF)

        return float(dF)

    def set_parameters(self, C: float = None, T: float = None, k: float = None):
        """
            set_parameters setter for the parameters T and k

        Parameters
        ----------
        T: float, optional
            Temperature in Kelvin, defaults to 398
        k: float, optional
            boltzmann Constant, defaults to const.k*const.Avogadro
        C: float, optional
            C is the initial guess of the free energy difference.

        """

        if isinstance(T, Number):
            self.constants.update({self.T: T})

        if isinstance(k, Number):
            self.constants.update({self.k: k})

        if isinstance(C, Number):
            self.constants.update({self.C: C})

        self._update_function()


# alternative class names
class bar(bennetAcceptanceRatio):
    pass
