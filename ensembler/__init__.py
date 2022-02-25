"""
Ensembler
Code to sample ensembles of 1D and 2D models with various algorithms.
"""

# Handle versioneer
from ._version import get_versions

# Add imports here
from .ensembler import *

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
