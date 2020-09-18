"""
    This file provides ensembler types, that are used to annotate and enforce certain types throughout the ensembler package.
"""

# Generic Types - provided to all other files from here
from typing import TypeVar, Union, List, Tuple, Iterable, Dict, NoReturn
from numbers import Number

# Dummy defs:
potentialCls = TypeVar("potential")
conditionCls = TypeVar("condition")
samplerCls = TypeVar("samplers")

systemCls = TypeVar("system")

ensembleCls = TypeVar("ensemble")

# Ensembler specific Types:
"""
from ensembler.potentials._baseclasses import _potentialCls 
potential = TypeVar("potential", bound=_potentialCls)

#define here dummy type, so it is useable in sub classes
system = TypeVar("system")  #dummyDef for samplers

from ensembler.conditions._conditions import _conditionCls
condition = TypeVar("condition", bound=_conditionCls)

from ensembler.samplers._basicIntegrators import _integratorCls
samplers = TypeVar("samplers", bound=_integratorCls)

from ensembler.system.basic_system import system
system = TypeVar("system", bound=system)
"""
