"""
Unit and regression test for the ensembler package.
"""

# Import package, test suite, and other packages as needed
import ensembler
import pytest
import sys

def test_ensembler_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ensembler" in sys.modules
