
![myimg](./EnsemblerLogo.png)

Welcome to Ensembler
==============================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/rinikerlab/ensembler/workflows/CI/badge.svg)](https://github.com/rinikerlab/ensembler/actions?query=branch%3Amaster+workflow%3ACI)
[![codecov](https://codecov.io/gh/rinikerlab/Ensembler/branch/master/graph/badge.svg)](https://codecov.io/gh/rinikerlab/Ensembler/branch/master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rinikerlab/Ensembler.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rinikerlab/Ensembler/context:python)
![Build package](https://github.com/rinikerlab/Ensembler/workflows/Python%20package/badge.svg)
[![Documentation](https://img.shields.io/badge/Documentation-here-white.svg)](https://rinikerlab.github.io/Ensembler/index.html)

## Description
Ensembler is a python package that allows fast and easy access to simulation of one and two-dimensional model systems.
It enables method development using small test systems and to deepen the understanding of a broad spectrum of molecular dynamics (MD) methods, starting from basic techniques to enhanced sampling and free energy calculations.
The ease of installing and using the package increases shareability, comparability, and reproducibility of scientific code developments.
Here, we provide insights into the package's implementation, its usage, and an application example for free energy calculation.

## Contents
The full Documentation can be found here:  https://rinikerlab.github.io/Ensembler/
### Potential functions

  Implement mathematical functions of interest for modelling purposes for esxample in chemistry.
  Implementation of new potentials is very easy, as there are only few functions that need to be overwritten.
  Implemented Potentials: Harmonic Oscillator, Wave function, etc.. 
  Also different dimensionalities can be used like 1D, 2D, and ND.

   You can try the code in your Browser here (or check out down below the Tutorial&Example section): 
   
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=example%2FTutorial_Potentials.ipynb)

### Samplers

   This module provides integrators for integrating potential functions. E.g. Monte Carlo, Velocity Verlet,...
   
### Systems

   This module is used to setup a simulation. It gets a potential, integrator and other parameters.

### Ensembles

   This module contains the implementation of the ConveyorBelt and will contain in future additional Replica Exchange approaches.

### Visualization

   This module contains predefined visualization a and animation functions.

### Analysis

   This module contains at the moment only Free Energy Calculations.

## How To Install
Download the git and use the setup.py script of the repository:

    cd /path/to/local/Ensembler/repo
    pyton setup.py install

Instead of install one could add the path to the Ensembler repository, if you want to use the code and modify it. 
The requirements need then to be installed as well like in the following examples.
  
   PIP:
    
    export PYTHONPATH=${PYTHONPATH}:/path/to/local/Ensembler/repo
    cd /path/to/local/Ensembler/repo
    pip install -r requirements.txt
    
   Anaconda:
   
    conda-develop /path/to/local/Ensembler/repo
    cd /path/to/local/Ensembler/repo
    conda create -n ensembler --file environment_unix.yml
    conda activate ensembler

For windows, we also provide the requirment files (requirements_unix.txt and environment_windows.yml).

Keep in mind: IDE's have their own awesome ways of doing this.

## Tutorials and Examples:

### Tutorials
Here we provide short introductions into how Potential functions can be used and sampled in simulations in Ensembler .
You can try the tutorials with Binder directly from your browser!

Potentials: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FTutorial_Potentials.ipynb)

Simulations: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FTutorial_Simulations.ipynb)

### Examples
Examples are advanced jupyter notebooks, covering a certain topic, going deeper into the methodology.

Enhanced Sampling: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FExample_EnhancedSampling.ipynb)

Free Energy Calculations: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FExample_FreeEnergyCalculationSimulation.ipynb)

Interactive ConveyorBelt: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FExample_ConveyorBelt.ipynb)

EDS-Potentials: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rinikerlab/Ensembler/master?filepath=examples%2FExample_EDS.ipynb)


## Authors

Benjamin J. Ries;
Stephanie M. Linker;
David F. Hahn

## Copyright

Copyright (c) 2020, Benjamin  J. Ries, Stephanie M. Linker, David F. Hahn


### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
