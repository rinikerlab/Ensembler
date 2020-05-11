from setuptools import setup

setup(
   name='Ensembler',
   version='1.0',
   description='This Package shall be  used to develop fast and efficient theoretic MD simulation tools',
   author='Benjamin Schroeder David Hahn',
   author_email='bschroed@ethz.ch',
   packages=['ensembler'],  #same as name
   install_requires=[
                     'typing',
                     'pandas',
                     'numpy',
                     'sympy',
                     'scipy',
                     'matplotlib'], #external packages as dependencies
)
