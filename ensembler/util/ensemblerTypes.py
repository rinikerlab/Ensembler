
"""
    This file provides ensembler types, that are used to annotate and enforce certain types throughout the ensembler package.
"""
from typing import TypeVar

#Generic Types
from typing import Tuple, List, Callable, Union
from numbers import Number

#Ensembpler specific Types:
from ensembler.potentials._baseclasses import _potentialCls 
potential = TypeVar("potential", bound=_potentialCls)


from ensembler.conditions._conditions import Condition
condition = TypeVar("condition", bound=Condition)


system = TypeVar("system")  #dummyDef for integrator
from ensembler.integrator._basicIntegrators import _integratorCls
integrator = TypeVar("integrator", bound=_integratorCls)

from ensembler.system import basic_system
print(vars(basic_system))

from ensembler.system.basic_system import system
system = TypeVar("system", bound=system)
