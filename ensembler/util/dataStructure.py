"""
Module: dataStructure
    This module shall be used to implement all needed data Structures for the project.
"""
from collections import namedtuple

import __main__

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

# make pickleable
setattr(__main__, basicState.__name__, basicState)
basicState.__module__ = "__main__"

setattr(__main__, lambdaState.__name__, lambdaState)
lambdaState.__module__ = "__main__"

setattr(__main__, envelopedPStstate.__name__, envelopedPStstate)
envelopedPStstate.__module__ = "__main__"
