"""
[summary]
"""

import numpy as np
from ensembler.ensemble import exchange_pattern
from ensembler.ensemble._replica_graph import _replicaExchange
from ensembler.util.ensemblerTypes import systemCls, Iterable, List, NoReturn


class temperatureReplicaExchange(_replicaExchange):
    """
    Temperature Replica Exchange is swapping the temperature frequently between replicas.
    This method was to our knowledge first descibed by Sugita and Okamoto in 1999
    """

    _parameter_name: str = "temperature"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1

    nSteps_between_trials: int

    def __init__(
        self,
        system: systemCls,
        temperature_range: Iterable = np.linspace(start=298, stop=320, num=3),
        exchange_criterium: callable = None,
        steps_between_trials: int = 20,
        exchange_trajs: bool = False,
    ):
        """
            __init__
                constructs an Ensemble that is exchanging the temperaturesbetween replicas

        Parameters
        ----------
        system: systemCls
            system class that is the basis of the ensemble
        temperature_range: Iterable, optional
            temperature range for replica graph (Default: np.linspace(start=298, stop=320, num=3))
        exchange_criterium: callable, optional
            exchange criterium (Default: None - metropolis)
        steps_between_trials: int, optional
            number of steps inbetween the trials. (Default: 20)
        exchange_trajs: bool, optional
            shall we exchange the trajectories (Default: False)
        """
        super().__init__(
            system=system,
            exchange_dimensions={self._parameter_name: temperature_range},
            exchange_criterium=exchange_criterium,
            steps_between_trials=steps_between_trials,
        )

        if exchange_trajs:
            self.exchange_param = "trajectory"
        else:
            self.exchange_param = "_currentPosition"

        self._exchange_pattern = exchange_pattern.localExchangeScheme(self)

    def _adapt_system_to_exchange_coordinate(self) -> NoReturn:
        """
        update the replica to the new coordinate set.

        """
        [self.replicas[replica]._update_current_vars_from_current_state() for replica in self.replicas]
        # self._scale_velocities_fitting_to_temperature(swapped_exCoord, original_exCoord)

    def _scale_velocities_fitting_to_temperature(self, original_T: List[float], swapped_T: List[float]) -> NoReturn:
        """
            adapt the velocities to the new coordinates

        Parameters
        ----------
        original_T:List[float]
            original temperatures
        swapped_T:List[float]
            swapped temperatures

        """
        if not any([getattr(self.replicas[replica], "_currentVelocities") is None for replica in self.replicas]):  # are there velocities?
            [
                setattr(
                    self.replicas[replica],
                    "_currentVelocities",
                    np.multiply(self.replicas[replica]._currentVelocities, np.divide(original_T[i], swapped_T[i])),
                )
                for i, replica in enumerate(self.replicas)
            ]


class HamiltonianReplicaExchange(_replicaExchange):
    pass


class replicaExchangeEnvelopingDistributionSampling(_replicaExchange):
    _parameter_name: str = "s"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1

    nSteps_between_trials: int

    def __init__(
        self,
        system: systemCls,
        s_range: Iterable = np.logspace(start=1, stop=-4, num=3),
        exchange_criterium=None,
        steps_between_trials=20,
        exchange_trajs: bool = False,
    ):
        """
            constructs a replic exchange enveloping distribution sampling (RE-EDS) ensemble. This approach was developed by Sidler, Schwaninger and Riniker 2016.
            It exchanges the smoothing parameter s during the simulations.

        Parameters
        ----------
        system: systemCls
            system class that is the basis of the ensemble
        s_range: Iterable, optional
            smootghin parameter range for eds in the replica graph (Default: np.linspace(start=1, stop=-4, num=3))
        exchange_criterium: callable, optional
            exchange criterium (Default: None - metropolis)
        steps_between_trials: int, optional
            number of steps inbetween the trials. (Default: 20)
        exchange_trajs: bool, optional
            shall we exchange the trajectories (Default: False)
        """
        super().__init__(
            system=system,
            exchange_dimensions={self._parameter_name: s_range},
            exchange_criterium=exchange_criterium,
            steps_between_trials=steps_between_trials,
        )

        if exchange_trajs:
            self.exchange_param = "trajectory"
        else:
            self.exchange_param = "_currentPosition"

        self._exchange_pattern = exchange_pattern.localExchangeScheme(self)

        # not needed?
        # for replicaID in self.replicas:
        #    replica = self.replicas[replicaID]
        #    #replica.set_s(replica.s_i)

    def _adapt_system_to_exchange_coordinate(self):
        """
        _adapt the system to the s-value change

        """
        for replicaID in self.replicas:
            replica = self.replicas[replicaID]
            replica._update_energies()
            replica.update_current_state()
