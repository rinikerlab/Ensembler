"""
Free Energy Calculations:
    This module contains functions for free energy calculations
    author: Gerhard König, Benjamin Schroeder
"""
#Generic Typing
from  decimal import Decimal

from numbers import Number
from typing import List, Dict, Tuple, Iterable

#Calculations
import numpy as np
import sympy as sp
from scipy import constants as const


class _FreeEnergyCalculator:
    constants: dict
    equation: sp.function
    simplified_equation: sp.function

    exp = np.vectorize(lambda x: x.exp())   #this function deals with decimals to
    J_to_cal:float = 0.239005736

    def __init__(self):
        pass

    def __str__(self):
        msg = ""
        msg += self.__class__.__name__+"\n"
        msg += "\tEquation: "+str(self.equation)+"\n"
        msg += "\tConstants:\n\t\t"+"\n\t\t".join([str(key)+":\t"+str(value) for key, value in self.constants.items()])+"\n"
        msg += "\tsimplified Equation: "+str(self.simplified_equation)+"\n"
        return msg


    def calculate(self, Vi:(Iterable[Number], Number), Vj:(Iterable[Number], Number))->float:
        raise NotImplementedError("This Function needs to be Implemented")

    
    def _update_function(self):
        self.simplified_equation = self.equation.subs(self.constants)

    @classmethod
    def _prepare_type(self, *arrays):
        return tuple(map(lambda arr: np.array(list(map(lambda x: Decimal(x), arr))), arrays))

    @classmethod
    def get_equation(cls)->sp.function:
        """
        get_equation returns the symoblic Equation
        
        :return: symbolic implementation of Zwanzig
        :rtype: sp.function
        """
        return cls.equation

    @classmethod
    def get_equation_simplified(cls)->sp.function:
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
    equation:sp.function = -(1/(k*T)) * sp.log(sp.exp(-(1/(k*T)) * (Vj-Vi)))
    constants:dict 

    def __init__(self, T:float=398, k:float=const.k * const.Avogadro, kT:bool=False, kJ:bool=False, kCal:bool=False):
        """
        __init__ Here you can set Class wide the parameters T and k for the Zwanzig Equation
        
        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """
        self.constants = {}
        if(kT):
            self.set_parameters(T=Decimal(1), k=Decimal(1))
        elif(kJ):
            self.set_parameters(T=Decimal(298), k=Decimal(const.k * const.Avogadro/1000))
        elif(kCal):
            self.set_parameters(T=Decimal(298), k=Decimal(const.k * const.Avogadro*self.J_to_cal/1000))
        else:
            self.set_parameters(T=Decimal(T), k=Decimal(k))
        self._update_function()


    def calculate(self, Vi:(Iterable[Number], Number), Vj:(Iterable[Number], Number))->float:
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
        return float(self._calculate_implementation_bruteForce(Vi, Vj))


    def _calculate_implementation_bruteForce(self, Vi:(Iterable, Number), Vj:(Iterable, Number))->float:
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
        if(not (len(Vi) == len(Vj))):
            raise ValueError("Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "+str(len(Vi)+" \t "+str(len(Vj)))+"\n")
       
        #Typcasting
        Vi, Vj = self._prepare_type(Vi, Vj)

        beta = 1/(self.constants[self.k]*self.constants[self.T])    #calculate Beta

        # Calculate the potential energy difference in reduced units of kT
        dVij = - beta * (Vj - Vi)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij = self.exp(dVij) 
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        #average of exponents
        meandVij = np.mean(edVij)

        # Return free energy difference
        try:
            #dF = zwanzigEquation.calculate(Vi, Vr) - zwanzigEquation.calculate(Vj, Vr) 
            dF = - (1/beta) * (meandVij).ln()
        except ValueError:
            raise ValueError("Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference "+str(np.mean(edVij[0:n-1])) )

        return dF

    def _calculate_meanEfficient(self, Vi:(Iterable, Number), Vj:(Iterable, Number))->float:
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
        if(not (len(Vi) == len(Vj))):
            raise ValueError("Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "+str(len(Vi)+" \t "+str(len(Vj)))+"\n")
        
        Vi, Vj = self._prepare_type(Vi, Vj)
        beta = 1/(cls.constants[cls.k]*cls.constants[cls.T])

        # Calculate the potential energy difference in reduced units of kT
        dVij = - beta * np.subtract(Vj, Vi)

        # Calculate offset for increased numerical stability
        meandVij = np.mean(dVij)

        # Exponentiate to obtain exp(-Delta U/kT)
        try:
            edVij= self.exp(dVij-meandVij)
        except OverflowError:
            raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

        # Return free energy difference
        try:
            dF = - (1/beta) * ((np.mean(edVij)).ln()+meandVij)
        except ValueError:
            raise ValueError("Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference "+str(np.mean(edVij)) )
        
        return dF

    def set_parameters(self, T:float=None, k:float=None):
        """
        set_parameters setter for the parameter
        
        :param T: Temperature in Kelvin, defaults to None
        :type T: float, optional
        :param k: Boltzmann Constant, defaults to None
        :type k: float, optional
        """
        if(isinstance(T, (Number, Decimal))):
            self.constants.update({self.T: Decimal(T)})

        if(isinstance(k, (Number, Decimal))):
            self.constants.update({self.k: Decimal(k)})
        self._update_function()

class zwanzig(zwanzigEquation):
    pass

class threeStateZwanzigReweighting(zwanzigEquation):
    k, T, Vi, Vj, Vr = sp.symbols("k T Vi Vj Vr")
    equation:sp.function = -(1/(k*T)) * (sp.log(sp.exp(-(1/(k*T)) * (Vi-Vr))) - sp.log(sp.exp(-(1/(k*T)) * (Vj-Vr))))

    def __init__(self,  kCal:bool=False, T:float=398, k:float=const.k * const.Avogadro, kT:bool=False, kJ:bool=False):
        """
        __init__ Here you can set Class wide the parameters T and k for the Zwanzig Equation
        
        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """
        super().__init__(T=T, k=k, kJ=kJ, kT=kT, kCal=kCal)

    def calculate(self, Vi:(Iterable[Number], Number), Vj:(Iterable[Number], Number), Vr:(Iterable[Number], Number))->float:
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
        return float(self._calculate_implementation_useZwanzig(Vi, Vj, Vr))

    def _calculate_implementation_useZwanzig(self, Vi:(Iterable, Number), Vj:(Iterable, Number), Vr:(Iterable[Number], Number))->float:
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
        if(not (len(Vi) == len(Vj) == len(Vr))):
            raise ValueError("Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "+str(len(Vi)+" \t "+str(len(Vj)))+"\n")
       
       #Type Casting
        Vi, Vj, Vr = self._prepare_type(Vi, Vj, Vr)

        #Build Zwanzig
        zwanz = zwanzigEquation()
        zwanz.constants = self.constants

        #Calc
        dF1r = zwanz.calculate(Vi, Vr)
        dF2r = zwanz.calculate(Vj, Vr)
        dF = dF2r - dF1r

        return dF

class dfEDS(threeStateZwanzigReweighting):
    pass

class bennetAcceptanceRatio(zwanzigEquation):
    k, T, C, Vi_i, Vj_i, Vi_j, Vj_j = sp.symbols("k T C  Vi_i Vj_i Vi_j Vj_j")
    equation:sp.function = (1/(k*T)) * (sp.log(sp.exp((1/(k*T)) * (Vi_j-Vj_j+C))) - sp.log(sp.exp((1/(k*T)) * (Vj_i-Vi_i+C))))
    constants:dict = {T: 298, k: const.k * const.Avogadro, C: Decimal(0)}

    #Numeric parameters
    convergence_radius:float = 0.005
    max_iterations:int = 1000
    min_iterations:int = 3

    def __init__(self, C:float=0.0, T:float=398, k:float=const.k * const.Avogadro,
                 kT:bool=False, kJ:bool=False, kCal:bool=False,
                convergence_radius:float=0.05, max_iterations:int=100, min_iterations:int=3):
        """
        __init__ Here you can set Class wide the parameters T and k for the bennet acceptance ration (BAR) Equation
        
        :param T: Temperature in Kelvin, defaults to 398
        :type T: float, optional
        :param k: boltzmann Constant, defaults to const.k*const.Avogadro
        :type k: float, optional
        """
        if(kT):
            self.set_parameters(T=Decimal(1), k=Decimal(1), C=C)
        elif(kJ):
            self.set_parameters(T=Decimal(298), k=Decimal(const.k * const.Avogadro/1000), C=C)
        elif(kCal):
            self.set_parameters(T=Decimal(298), k=Decimal(const.k * const.Avogadro*self.J_to_cal/1000), C=C)
        else:
            self.set_parameters(T=T, k=k, C=C)

        #deal with numeric params:
        self.convergence_radius = convergence_radius
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        self._update_function()

    def calculate(self, Vi_i:(Iterable[Number], Number), Vj_i:(Iterable[Number], Number),
                        Vi_j:(Iterable[Number], Number), Vj_j:(Iterable[Number], Number))->float:
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
        return self._calculate_meanEfficient(Vi_i, Vj_i, Vi_j, Vj_j)

    
    def _calculate_meanEfficient(self, Vi_i:(Iterable[Number], Number), Vj_i:(Iterable[Number], Number),
                                 Vi_j:(Iterable[Number], Number), Vj_j:(Iterable[Number], Number))->float:
        """

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
        if(not (len(Vi_i) == len(Vi_i) == len(Vi_j)== len(Vj_j))):  #I and J simulation don't need the same length.
            raise ValueError("Zwanzig Error: The given arrays for Vi and Vj must have the same length. \n Actually they have: "+str(len(Vi_i)+" \t "+str(len(Vj_i)))+"\n"+str(len(Vi_j)+" \t "+str(len(Vj_j)))+"\n")
       
        #Type Cast
        Vi_i, Vj_i, Vi_j, Vj_j = self._prepare_type(Vi_i, Vj_i, Vi_j, Vj_j)

        #Calc Beta
        beta = Decimal(1/(self.constants[self.k]*self.constants[self.T]))

        iteration_diff = 1
        iteration = 0
        dF:Decimal=Decimal(0)
        while (iteration_diff > self.convergence_radius or  self.min_iterations > iteration) and self.max_iterations > iteration:
                
            # Calculate the potential energy difference in reduced units of kT
            dV_i = beta * ((Vj_i-Vi_i) + self.constants[self.C])
            dV_j = beta * ((Vi_j-Vj_j) - self.constants[self.C])

            # Exponentiate to obtain exp(-Delta U/kT)
            try:
                edV_i=self.exp(dV_i)
                edV_j=self.exp(dV_j)
            except OverflowError:
                raise OverflowError("Zwanzig Error: Overflow in exponentiation of potential energy difference. Aborting calculation.")

            #get average
            mean_edV_i = np.mean(edV_i)
            mean_edV_j = np.mean(edV_j)

\            # Return free energy difference
            try:
                dF = (1/beta) * (mean_edV_j / mean_edV_i).ln()
            except ValueError:
                raise ValueError("Zwanzig Error: Problems taking logarithm of the average exponentiated potential energy difference "+str(np.mean(edVij[0:n-1])) )
            
            iteration_diff = abs(float(self.constants[self.C]-dF))
            iteration += 1

            self.constants.update({self.C: (self.constants[self.C]+dF)/2})  #no averaging?
            print("Iteration: "+str(iteration)+"\tdF: "+str(dF)+"\t\ttDiff: ", iteration_diff)


        if(iteration >= self.max_iterations):
            raise Exception("BAR is not converged after "+str(iteration)+" steps. stopped at: "+str(dF))
        print("Final Iterations: ", iteration, " Result: ", dF, "\t", iteration_diff)

        return float(dF)

    def set_parameters(self, C:float=None, T:float=None, k:float=None):
        """
        set_parameters setter for the parameter
        
        :param T: Temperature in Kelvin, defaults to None
        :type T: float, optional
        :param k: Boltzmann Constant, defaults to None
        :type k: float, optional
        """
        if(isinstance(T, Number)):
            self.constants.update({self.T: Decimal(T)})

        if(isinstance(k, Number)):
            self.constants.update({self.k: Decimal(k)})

        if(isinstance(C, Number)):
            self.constants.update({self.C: Decimal(C)})

        self._update_function()


#alternative class names
class bar(bennetAcceptanceRatio):
    pass



"""
 
#Suff
from itertools import combinations

# eds_zwanzig
def calc_eds_zwanzig(V_is: List[Iterable], V_R: Iterable, undersampling_tresh: int = 0, temperature: float = 298, verbose: bool = False) -> (Dict[Tuple[int, int], float], List[Tuple[float, float]]):
        .. autofunction:: calculate_EDS_Zwanzig
        implementation from gromos++
        This function is calculating the relative Free Energy of multiple
        punish non sampling with "sampled mean"
        :param V_is:

        :warning: needs testing
    # Todo: check same length! and shape
    #TODO TEST
    # print(V_is[0].shape)
    # Todo: parse Iterables to np arrays or pandas.series

    if (verbose): print("params:")
    if (verbose): print("\ttemperature: ", temperature)
    if (verbose): print("\tundersampling_tresh: ", undersampling_tresh)

    # calculate:
    ##first exponent
    ##rowise -(V_i[t]-V_R[t]/(kb*temperature))
    exponent_vals = [np.multiply((-1 / (c.k * temperature)), (np.subtract(V_i, V_R))) for V_i in V_is]
    exponent_vals = [V_i.apply(lambda x: x if (x != 0) else 0.00000000000000000000000001) for V_i in exponent_vals]

    ##calculate dF_i = avgerage(ln(exp(exponent_vals)))
    state_is_sampled = [[True if (t < undersampling_tresh) else False for t in V_i] for V_i in V_is]
    check_sampling = lambda sampled, state: float(np.std([V for (sam, V) in zip(sampled, state) if (sam)]))
    std_sampled = []
    for sampled, state in zip(state_is_sampled, V_is):
        std_sampled.append(check_sampling(sampled, state))

    sampling = [np.divide(np.sum(sam), len(sam)) for sam in state_is_sampled]
    extrema = [(np.min(np.log(np.exp(column))), np.max(np.log(np.exp(column)))) for column in exponent_vals]

    dF_t = [np.log(np.exp(column)) for column in exponent_vals]
    dF = {column.name: {"mean": np.mean(column), "std": np.std(column), "sampling": sampled_state, "std_sampled": std_sampl} for
          column, sampled_state, std_sampl in zip(dF_t, sampling, std_sampled)}

    if (verbose): print("\nDf[stateI]: (Mean\tSTD\tsampling)")
    if (verbose): print(
        "\n".join([str(stateI) + "\t" + str(dFI["mean"]) + "\t" + str(dFI["std"]) + "\t" + str(dFI["sampling"]) for stateI, dFI in dF.items()]))

    ##calculate ddF_ij and estimate error with Gauss
    ddF = {tuple(sorted([stateI, stateJ])): {"ddF": np.subtract(dF[stateI]["mean"], dF[stateJ]["mean"]), "gaussErrEstmSampled": np.sqrt(
        np.square(dF[stateI]["std_sampled"]) + np.square(dF[stateJ]["std_sampled"])),
                                             "gaussErrEstm": np.sqrt(np.square(dF[stateI]["std"]) + np.square(dF[stateJ]["std"]))} for stateI, stateJ
           in combinations(dF, r=2)}
    if (verbose): print("\nDDF[stateI][stateJ]\t diff \t gaussErrEstm")
    if (verbose): print(
        "\n\t".join([str(stateIJ) + ": " + str(ddFIJ["ddF"]) + " +- " + str(ddFIJ["gaussErrEstm"]) for stateIJ, ddFIJ in ddF.items()]))
    if (verbose): print()
    return ddF, dF
"""
 
 