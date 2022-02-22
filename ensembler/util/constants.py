from rdkit import Chem
from collections import namedtuple
from scipy import constants as const
from scipy.constants import physical_constants as pc

from ensembler.util.units import nm, kJ

atom_masses = {"C": pc}
#phys constants
k = const.k


#Atoms - Elements
pt = Chem.GetPeriodicTable()
element_information = namedtuple("element", ["weight", "atomNumber", "symbol", "vdwRadius", "covRadius", "valence"])
elements = {pt.GetElementSymbol(atomID):element_information(weight=pt.GetAtomicWeight(atomID),
                                                 atomNumber=atomID,
                                                 symbol=pt.GetElementSymbol(atomID),
                                                 vdwRadius=pt.GetRvdw(atomID),
                                                 covRadius=pt.GetRcovalent(atomID),
                                                 valence=pt.GetDefaultValence(atomID))
                                                 for atomID in range(118)}

#Covalent Bonds
bond_information = namedtuple("bond", ["r0", "k_harm"])

##C-C
k_harm_CC = 7.1500000e+06 * (kJ/nm**2)
k_quart_CC = 3.3474870e+05 * (kJ/nm**4)
r_0_CC = 1.5300000e-01*nm

bonds = {"C":{"C": bond_information(r0=r_0_CC, k_harm=k_harm_CC)}}