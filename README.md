Ensembler
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/ensembler/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/ensembler/actions?query=branch%3Amaster+workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Ensembler/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/Ensembler/branch/master)


Code to sample ensembles of simple (toy) models with various algorithms. 

## Description
This project tries to give users very easy to use and simple functionality to develop code for physical ensembles.
 
## Contents
### Potential functions

  Contains simple functions, that can be stocked together. 
  Also implementation of new potentials is very easy, as there only few functions that need to be overwritten.
  Examples: Harmonic Oscillator, Wa*ve function, etc.. 
  Also different dimensionalities can be used.

   * OneD

   * TwoD

   * ND

### Systems

   This module is used to setup a simulation. It gets a potential, integrator and other parameters.

### Integrators

   This module provides integrators for integrating potential functions. E.g. Monte Carlo, Velocity Verlet,...

### Visualization

   This module contains predefined visualization a and animation functions.

## How To Install




## Authors

Benjamin Schroeder;
Stephanie Linker;
David Hahn

## Copyright

Copyright (c) 2020, Benjamin Schröder, Stephanie Linker, David Hahn


### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
