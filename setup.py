from setuptools import setup

setup(
   name='Ensembler',
   version='1.0',
   description='This Package shall be a tool for fast and efficient development of theoretic thermodynamic simulation tools or teaching.',
   author='Benjamin Schroeder; David Hahn',
   author_email='bschroed@ethz.ch',
   packages=['ensembler'],  #same as name
   install_requires=[
                     'typing',
                     'pandas',
                     'numpy',
                     'sympy',
                     'scipy',
                     'tqdm',
                     'ipywidgets',
                     'matplotlib'], #external packages as dependencies
)
