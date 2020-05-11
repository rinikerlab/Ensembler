"""
Module: dataStructure
    This module shall be used to implement all needed data Structures for the project.
"""
from collections import namedtuple

#states:
basicState = namedtuple("State", ["position", "temperature",
                                  "totEnergy", "totPotEnergy", "totKinEnergy",
                                  "dhdpos", "velocity"])

lambdaState = namedtuple("Lambda_State", ["position", "temperature",
                                   "totEnergy", "totPotEnergy", "totKinEnergy",
                                   "dhdpos",  "velocity",
                                   "lam", "dhdlam"])

envelopedPStstate = namedtuple("EDS_State", ["position", "temperature",
                                         "totEnergy", "totPotEnergy", "totKinEnergy",
                                         "dhdpos",  "velocity",
                                         "s", "Eoff"])
