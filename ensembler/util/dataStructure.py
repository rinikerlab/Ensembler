"""
Module: dataStructure
    This module contains all needed data Structures for the project.
"""
from collections import namedtuple

import __main__

"""
States
    States are used by systems, to represent a state of the system, by a collection of variables. 
    For example is the current state defined by a state structure.
    
    The states also define the variables contained in a trajectory.
"""
# states:
basicState = namedtuple("State", ["position", "temperature",
                                  "totEnergy", "totPotEnergy", "totKinEnergy",
                                  "dhdpos", "velocity"])

lambdaState = namedtuple("Lambda_State", ["position", "temperature",
                                          "totEnergy", "totPotEnergy", "totKinEnergy",
                                          "dhdpos", "velocity",
                                          "lam", "dhdlam"])

envelopedPStstate = namedtuple("EDS_State", ["position", "temperature",
                                             "totEnergy", "totPotEnergy", "totKinEnergy",
                                             "dhdpos", "velocity",
                                             "s", "Eoff"])


# make states pickle-able
setattr(__main__, basicState.__name__, basicState)
basicState.__module__ = "__main__"

setattr(__main__, lambdaState.__name__, lambdaState)
lambdaState.__module__ = "__main__"

setattr(__main__, envelopedPStstate.__name__, envelopedPStstate)
envelopedPStstate.__module__ = "__main__"

