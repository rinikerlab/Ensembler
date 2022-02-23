# Disable Pint's old fallback behavior (must come before importing Pint)
import os
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = "0"

import pint
from pint.quantity import _Quantity

from sympy import sympify
_Quantity._sympy_ = lambda s: sympify(f'{s.m}*{s.u:~}')

#Build a unitRegistry
unit_registry = pint.UnitRegistry()
unit_registry.default_format = '~'
quantity = unit_registry.Quantity


#Recommended Units
## Energies
kJ = unit_registry.kJ
kcal = unit_registry.kcal
kb = unit_registry.k

## Temperature
K = unit_registry.K

## Geometrie: Distance
nm = unit_registry.nm
a = unit_registry.angstrom
deg = unit_registry.degree


C = unit_registry.C

#Suggested Constants:
k_harm = 1 * kJ / nm #used in gromos