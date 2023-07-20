from ensembler.potentials.OneD import harmonicOscillatorPotential

from ensembler.util import constants
from ensembler.util import units


class bondTerm(harmonicOscillatorPotential):
    """
    Todo: nice docstring here
    """

    def __init__(self, k: float = constants.k_harm_CC, r_0: float = constants.r_0_CC, V_0: float = 0.0 * units.kJ):
        super().__init__(k=k, x_shift=r_0, y_shift=V_0, unitless=False)
