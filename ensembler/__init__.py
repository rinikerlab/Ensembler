"""
Ensembler
Code to sample ensembles of simple (toy) models with various algorithms. 
"""

# Add imports here
from .ensembler import *
from .ensembler.util import ensemblerTypes

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
