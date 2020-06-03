# Ensembler


__CURRENTLY UNDER CONSTRUCTION __


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

   You can try the code in your Browser here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/improving_import_structure?filepath=examples%2FBasicPotentials.ipynb)

### Systems

   This module is used to setup a simulation. It gets a potential, integrator and other parameters.

### Integrators

   This module provides integrators for integrating potential functions. E.g. Monte Carlo, Velocity Verlet,...

### Ensembles

   This module contains the implementation of the ConveyorBelt and will contain in future additional Replica Exchange approaches.

### Visualization

   This module contains predefined visualization a and animation functions.

### Analysis

   This module contains at the moment only Free Energy Calculations.

## How To Install
If you want to use the package you can either install the package with pip:

   ' pip install Ensembler '

Or you can also add the path to the Ensembler repository, if you want to use the code and modify it.
   UNIX:
   'export PYTHONPATH=${PYTHONPATH}:/path/to/local/Ensembler/repo'

   IDE's have their own way of doing this.

## Examples:

### Tutorials

Potentials: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/improving_import_structure?filepath=examples%2FBasicPotentials.ipynb)

Simulations: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/improving_import_structure?filepath=examples%2FBasicSimulations.ipynb)

Interactive ConveyorBelt: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/improving_import_structure?filepath=examples%2FConveyorBelt.ipynb)

EDS-Potentials: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/improving_import_structure?filepath=examples%2FEDS.ipynb)

Free Energy Calculations: __Under Construction__


### Authors

Benjamin Schroeder;
David Hahn
