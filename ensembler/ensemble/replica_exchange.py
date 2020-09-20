"""
[summary]
"""

from collections import Iterable

import numpy as np

from ensembler.ensemble import exchange_pattern
from ensembler.ensemble._replica_graph import ReplicaExchange


class temperatureReplicaExchange(ReplicaExchange):
    _parameter_name: str = "temperature"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1

    nSteps_between_trials: int

    def __init__(self, system, temperature_Range: Iterable = np.linspace(start=298, stop=320, num=3),
                 exchange_criterium=None, steps_between_trials=20,
                 exchange_trajs: bool = False):
        super().__init__(system=system, exchange_dimensions={self._parameter_name: temperature_Range},
                         exchange_criterium=exchange_criterium, steps_between_trials=steps_between_trials)

        if (exchange_trajs):
            self.exchange_param = "trajectory"
        else:
            self.exchange_param = "_currentPosition"

        self._exchange_pattern = exchange_pattern.localExchangeScheme(self)

    def _adapt_system_to_exchange_coordinate(self, swapped_exchange_coord, original_exchange_coord):
        [self.replicas[replica]._update_current_vars_from_current_state() for replica in self.replicas]
        # self._scale_velocities_fitting_to_temperature(swapped_exCoord, original_exCoord)

    def _scale_velocities_fitting_to_temperature(self, original_T, swapped_T):
        if (not any([getattr(self.replicas[replica], "_currentVelocities") is None for replica in
                     self.replicas])):  # are there velocities?
            [setattr(self.replicas[replica], "_currentVelocities",
                     np.multiply(self.replicas[replica]._currentVelocities, np.divide(original_T[i], swapped_T[i]))) for
             i, replica in enumerate(self.replicas)]


class HamiltonianReplicaExchange(ReplicaExchange):
    pass


class replicaExchangeEnvelopingDistributionSampling(ReplicaExchange):
    _parameter_name: str = "s"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1

    nSteps_between_trials: int

    def __init__(self, system, s_range: Iterable = np.logspace(start=1, stop=-4, num=3), exchange_criterium=None,
                 steps_between_trials=20,
                 exchange_trajs: bool = False):
        super().__init__(system=system, exchange_dimensions={self._parameter_name: s_range},
                         exchange_criterium=exchange_criterium, steps_between_trials=steps_between_trials)

        if (exchange_trajs):
            self.exchange_param = "trajectory"
        else:
            self.exchange_param = "_currentPosition"

        self._exchange_pattern = exchange_pattern.localExchangeScheme(self)

        # not needed?
        # for replicaID in self.replicas:
        #    replica = self.replicas[replicaID]
        #    #replica.set_s(replica.s_i)

    def _adapt_system_to_exchange_coordinate(self, swapped_exchange_coord, original_exchange_coord):
        for replicaID in self.replicas:
            replica = self.replicas[replicaID]
            replica._update_energies()
            replica.update_current_state()
