"""
Replica Graph
    this Module contains two files
"""
import copy
import warnings
import itertools as it
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.constants as const

from tqdm import tqdm

import multiprocessing as mult

from ensembler.ensemble import exchange_pattern
from ensembler.util.basic_class import _baseClass
from ensembler.util.ensemblerTypes import systemCls, List, Dict, Tuple, Iterable, Union, NoReturn, Number


class _mutliReplicaApproach(_baseClass):
    """
    This base class is a scaffold for simulations processes containing multiple system replicas.
    """

    ##Parameters - to build replica graphs
    ### EDGES - Exchange parameters - gradients
    coordinate_dimensions: int
    parameter_names: List
    exchange_dimensions: Dict

    ### NODES - Replicas

    """
    Attributes
    """

    @property
    def replicas(self) -> Dict[int, systemCls]:
        return self._replicas

    @property
    def nReplicas(self) -> int:
        return self._nReplicas

    @property
    def replica_graph_dimensions(self) -> int:
        return self._replica_graph_dimensions

    @property
    def replica_trajectories(self) -> Dict[int, pd.DataFrame]:
        """
            access all replica_trajectories
        Returns
        -------
        Dict[int, pd.DataFrame]
            dictionary containing the unique identifier of the replica as key and the trajectory
        """
        return {coord: replica.trajectory for coord, replica in self.replicas.items()}

    def get_trajectories(self) -> Dict[int, pd.DataFrame]:
        """
            get all trajectories of the replica graph.

        Returns
        -------
        Dict[int, pd.DataFrame]
            dictionary containing the unique identifier of the replica as key and the trajectory
        """
        return self.replica_trajectories

    @property
    def replica_positions(self) -> Dict[int, Union[Number, Iterable[Number]]]:
        """
            all replica positions

        Returns
        -------
         Dict[int, Union[Number, Iterable[Number]]]
            returns dictionary with replica unique key and the current positions
        """
        return {replicaName: replica.position for replicaName, replica in self.replicas.items()}

    @replica_positions.setter
    def replica_position(self, positions: Dict[int, Union[Number, Iterable[Number]]]) -> NoReturn:
        """
            change the current position of all replicas

        Parameters
        ----------
        positions: Dict[int, Union[Number, Iterable[Number]]]
            new current position (Number) with the unique identifier (int) of the replicas

        """
        if len(positions) == self.nReplicas:
            if isinstance(positions, dict):
                for replicaName, position in positions.items():
                    self.replicas[replicaName].position = position
            else:
                raise Exception("Did not understand the the type of the new positions " + str(type(positions)))
        else:
            raise ValueError(
                "Not enough positions got passed to setReplicapositions!\n replicas: "
                + str(self.nReplicas)
                + "\n positions: "
                + str(len(positions))
                + "\n"
                + str(positions)
            )

    def get_replicas_positions(self) -> Dict[int, Union[Number, Iterable[Number]]]:
        """
        getReplicaPositions
            get all replica current positions
        Returns
        -------
        Dict[int, Union[Number, Iterable[Number]]]
            all replica positions
        """
        return self.replica_positions

    def set_replicas_positions(self, positions: Dict[int, Union[Number, Iterable[Number]]]) -> NoReturn:
        """
            set new current positions to all replicas
        Parameters
        ----------
        positions: Dict[int, Union[Number, Iterable[Number]]]
            new positions for all replicas
        """

        self.replica_position = positions

    @property
    def replica_velocities(self) -> Dict[int, Union[Number, Iterable[Number]]]:
        """
            Access all replica velocities

        Returns
        -------
        Dict[int, Number]
            unique key(int) with velocities(Number)
        """
        return {replicaName: getattr(replica, "_currentVelocities") for replicaName, replica in self.replicas.items()}

    @replica_velocities.setter
    def replica_velocities(self, velocities: Dict[int, Union[Number, Iterable[Number]]]) -> NoReturn:
        """
            set new velocities for all replicas.

        Parameters
        ----------
        velocities: Dict[int, Union[Number, Iterable[Number]]]
            new velocities

        Returns
        -------

        """
        if len(velocities) == self.nReplicas:
            if isinstance(velocities, dict):
                for replicaName, velocity in velocities.items():
                    self.replicas[replicaName].set_velocities(velocity)
            else:
                raise Exception("Did not understand the the type of the new positions " + str(type(velocities)))
        else:
            raise ValueError(
                "Not enough positions got passed to setReplicapositions\n replicas: "
                + str(self.nReplicas)
                + "\n positions: "
                + str(len(velocities))
                + "\n"
                + str(velocities)
            )

    def get_replicas_velocities(self) -> Dict[int, Union[Number, Iterable[Number]]]:
        """
            get all replica current velocities

        Returns
        -------
        Dict[int,  Union[Number, Iterable[Number]]]
            all current_velocities mapped on coordinate
        """

        return self.replica_velocities

    def set_replicas_velocities(self, velocities: Dict[int, Union[Number, Iterable[Number]]]) -> NoReturn:
        """

            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new velocity.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)

        Parameters
        ----------
        velocities: Dict[int, Union[Number, Iterable[Number]]]
            new velocities

        """
        self.replica_velocities = velocities

    @property
    def replica_current_states(self) -> Dict[int, namedtuple]:
        """
            returns all current_states of all replicas

        Returns
        -------
        Dict[int, namedtuple]
            unique identifier of replica mapped on current state

        """
        return {replicaName: replica.current_state for replicaName, replica in self.replicas.items()}

    def get_replicas_current_states(self) -> Dict[int, namedtuple]:
        """
            get all current states of the replicas.

        Returns
        -------
        Dict[Number, namedtuple]
            all current states mapped on replica coordinates
        """
        return self.replica_current_states

    def get_replica_total_energies(self) -> Dict[int, float]:
        """
            get all current total system energies of the replicas

        Returns
        -------
        Dict[Tuple, float]
            total current system energies mapped on replica coordinates
        """
        [replica._update_energies() for coord, replica in self.replicas.items()]
        return {coord: replica.total_system_energy for coord, replica in self.replicas.items()}

    def set_parameter_set(self, coordinates: List, replica_indices: List) -> NoReturn:
        """
            set ParameterSet
            This function is setting new coordinates to the replicas in the replica lists.
            The coordinates will be assigned sequentially in the same order to the replicas List.
        Parameters
        ----------
        coordinates:List
            list of coordinates

        replica_indices: List
            List of replicas (in order like the coordinates)

        Returns
        -------

        """

        if self.coordinate_dimensions > 1:
            for coords, replica in zip(coordinates, replica_indices):
                for ind, parameter_Name in enumerate(self.exchange_dimensions):
                    parameters = list(self.exchange_dimensions[self.parameter_Name])
                    # set parameter set
                    if hasattr(self.replicas[replica], parameter_Name):
                        setattr(self.replicas[replica], parameter_Name, parameters[coords[ind]])
                    else:
                        raise Exception("REPLICA INIT FAILDE: Replica does not have a field: " + parameter_Name + "\n")
        else:
            parameters = list(self.exchange_dimensions[self.parameter_names[0]])

            for coords, replica in zip(coordinates, replica_indices):
                # set parameter set
                if hasattr(self.replicas[replica], self.parameter_names[0]):
                    setattr(self.replicas[replica], self.parameter_names[0], parameters[coords])

                else:
                    raise Exception("REPLICA INIT FAILDE: Replica does not have a field: " + self.exchange_dimensions[0] + "\n")

    # const
    def __init__(self):
        self._replicas: dict = {}
        self._nReplicas: int = None
        self._replica_graph_dimensions: int = None

    ##init funcs
    def _initialise_replica_graph(self, verbose: bool = False) -> NoReturn:
        """
            _initialise_replica_graph
                initialize the underlying replica graph by generating all nodes, fitting to all parameter combinations.

        Parameters
        ----------
        verbose: bool, optional
            Make some noise! (default: False)

        """
        coordinates = self._generate_all_node_coordinates_combinations()

        if verbose:
            print("\nBUILD Replicas")
            print("Coordinates\n\t", coordinates, "\n")

        self._build_nodes(coordinates=coordinates)

        if verbose:
            print(
                "Replicas:\n\treplicaID\tcoordinates\n\t"
                + "\n\t".join([str(self.replicas[x].uniqueID) + "\t" + str(self.replicas[x].Nodecoordinates) for x in self.replicas])
            )

    ### * Node Functions
    #### Generate coordinates
    def _generate_all_node_coordinates_combinations(self) -> List[namedtuple]:
        """
            _generate_all_node_coordinates_combinations
                generate all node coordinates of the replica graph
        Returns
        -------
        List[namedtuple]
            the list of all coordinates to be used
        """
        self.coord_dims = list(sorted(self.exchange_dimensions))
        self.coord_names = list(sorted(self.exchange_dimensions.keys()))
        self.coord_set = namedtuple("coordinates", self.coord_names)

        # make pickleable
        import __main__

        setattr(__main__, self.coord_set.__name__, self.coord_set)
        self.coord_set.__module__ = "__main__"

        # generate all parameter combinations
        if len(self.exchange_dimensions) > 1:
            coord_it = list(it.product(*[list(self.exchange_dimensions[r]) for r in sorted(self.exchange_dimensions)]))
            coordinates = [self.coord_set(**{name: x for name, x in zip(self.coord_names, x)}) for x in coord_it]
        elif len(self.exchange_dimensions) == 1:
            coordinates = list(map(lambda x: self.coord_set(x), self.exchange_dimensions[self.coord_dims[0]]))
        else:
            raise Exception("Could not find parameters to exchange")

        return coordinates

    ###build nodes
    def _build_nodes(self, coordinates: Dict[str, any]) -> NoReturn:
        """
            _build_nodes
                use given default system and modify the exchange coordinates in order to generate all nodes

        Parameters
        ----------
        coordinates: Dict[str, any]
            node coordinates for the Replica graph

        """
        ## deepcopy all
        self._nReplicas = len(list(coordinates))

        # for x in range(self.nReplicas):
        #    print("COPY!", x)
        #    copy.deepcopy(self.system)
        replicas = [copy.deepcopy(self.system) for x in range(self.nReplicas)]  # generate deepcopies

        # build up graph - set parameters
        replicaID = 0
        for coords, replica in zip(coordinates, replicas):

            ## finalize deepcopy:
            # replica.trajectory = [] #fields are not deepcopied!!!
            replica.nsteps = self.nSteps_between_trials  # set steps between trials

            ## Node properties
            setattr(replica, "replicaID", replicaID)
            setattr(replica, "Nodecoordinates", coords)

            ##set parameters
            parameter_set = {}
            for ind, parameter_Name in enumerate(self.coord_dims):
                if hasattr(replica, parameter_Name) or True:
                    if isinstance(coords, Iterable):
                        setattr(replica, parameter_Name, coords[ind])
                    else:
                        setattr(replica, parameter_Name, coords)
                else:
                    raise Exception("REPLICA INIT FAILED: Replica does not have a field: " + parameter_Name + "\n")
                parameter_set.update({parameter_Name: coords[ind]})

            setattr(replica, "exchange_parameters", parameter_set)
            replica._init_velocities()  # e.g. if temperature changes

            self.replicas.update({replicaID: replica})
            replicaID += 1


class _replicaExchange(_mutliReplicaApproach):
    """
    This base class is build on multi Replica approach and provides the exchange functionality scaffold for the nodes in the replica graph.

    """

    ##Exchange Variables
    _currentTrial: int
    _exchange_pattern: exchange_pattern.Exchange_pattern = None
    nSteps_between_trials: int

    ###METROPOLIS CRITERION
    ###default Metropolis Criterion
    _default_metropolis_criterion = lambda self, originalParams, swappedParams: (
        np.greater_equal(originalParams, swappedParams) or self._default_randomness(originalParams, swappedParams)
    )
    exchange_criterium = _default_metropolis_criterion

    ###random part of Metropolis Criterion:
    _randomness_factor = 0.1
    _temperature_exchange: float = 298
    _default_randomness = lambda self, originalParams, swappedParams: (
        (1 / self._randomness_factor) * np.random.rand()
        <= np.exp(-1.0 / (const.gas_constant / 1000.0 * self._temperature_exchange) * (originalParams - swappedParams + 0.0000001))
    )  # pseudo count, if params are equal

    def __init__(
        self,
        system: systemCls,
        exchange_dimensions: Dict[str, Iterable],
        exchange_criterium=_default_metropolis_criterion,
        steps_between_trials: int = 10,
    ):
        """
            __init__
                builds the exchange replica scaffold.
                - Builds up the replica Grap (via _mutliReplicaApproach
                - Builds up exchange mechanisms.

        Parameters
        ----------
        system: systemCls
            the default system for the Replica Exchange
        exchange_dimensions: Dict[str, Iterable]
            exchange dimensions
        exchange_criterium: lambdaFunction, optional
            criterium basis to exchange the nodes
        steps_between_trials: int, optional
            number of steps between exchange trials
        """
        super().__init__()

        # SET PARAMETER FIELDS
        if isinstance(exchange_dimensions, dict):
            self.exchange_dimensions = exchange_dimensions
        self.coordinate_dimensions = len(exchange_dimensions)  # get dimensionality
        self.parameter_names = list(self.exchange_dimensions.keys())

        # SET SYSTEM
        self.system = system

        # exchange finfo:
        self._exchange_information = pd.DataFrame(
            columns=[
                "nExchange",
                "uniqueReplicaID",
                "replicaI",
                "exchangeCoordinateI",
                "TotEI",
                "replicaJ",
                "exchangeCoordinateJ",
                "TotEJ",
                "doExchange",
            ]
        )

        if steps_between_trials is not None:
            self.nSteps_between_trials = steps_between_trials

        # initialize the replica graphs
        self.initialise()

        # exchange Criterium
        if exchange_criterium != None:
            self.exchange_criterium = exchange_criterium

        if isinstance(self._exchange_pattern, type(None)):
            self._exchange_pattern = exchange_pattern.localExchangeScheme(replica_graph=self)

        # Exchange params/io
        self._exchange_information: pd.DataFrame = pd.DataFrame(
            columns=[
                "nExchange",
                "replicaID",
                "replicaPositionI",
                "exchangeCoordinateI",
                "TotEI",
                "replicaPositionJ",
                "exchangeCoordinateJ",
                "TotEJ",
                "doExchange",
            ]
        )

    # public functions
    def initialise(self) -> NoReturn:
        """
        initialises the replica scaffold. (exchange and replica graph structure)
        """
        self._currentTrial = 0
        # BUILD replicas
        self._initialise_replica_graph()
        self._init_exchanges()

    def simulate(self, ntrials: int, steps_between_trials: int = None, reset_ensemble: bool = False) -> Dict[str, namedtuple]:
        """
        simulates the replica exchange approach by executing ntrials with x steps between the trials.

        Parameters
        ----------
        ntrials: int
            number of exchange trials
        steps_between_trials: int, optional
            steps between the exchange trials (Default: None - use object attribute value)
        reset_ensemble: bool,  optional
            reset the ensemble to start (default: false)

        Returns
        -------
        Dict[str, namedtuple]
            returns the current state of all replicas
        """
        if reset_ensemble:
            self._currentTrial = 0
            [replica.initialise(withdraw_Traj=True) for repName, replica in self.replicas.items()]
            self._exchange_information = pd.DataFrame(
                columns=[
                    "nExchange",
                    "replicaID",
                    "replicaPositionI",
                    "exchangeCoordinateI",
                    "TotEI",
                    "replicaPositionJ",
                    "exchangeCoordinateJ",
                    "TotEJ",
                    "doExchange",
                ]
            )
            self._init_exchanges()

        if isinstance(steps_between_trials, int):
            self.set_simulation_steps_between_trials(nsteps=steps_between_trials)

        for _ in tqdm(range(ntrials), desc="Running trials", leave=True):
            self.run()
            self.exchange()
        return self.get_replicas_current_states()

    def exchange(self) -> NoReturn:
        """
        Try to exchange the replica nodes according to the exchange pattern.
        """
        self._exchange_pattern.exchange(verbose=self.verbose)

    def run(self, verbosity: bool = False) -> NoReturn:
        """
            run simulation for all replicas

        Parameters
        ----------
        verbosity: bool, optional
            MORE Output!

        """
        for replica_coords, replica in self.replicas.items():
            replica.simulate(steps=self.nSteps_between_trials, withdraw_traj=False, init_system=False, verbosity=verbosity)

    def _run_parallel(self, verbosity: bool = False, nProcesses: int = 4) -> NoReturn:
        """
         @TODO - NOT IMPLEMENTED
        Parameters
        ----------
        verbosity
        nProcesses

        Returns
        -------

        """
        """this is an ugly work around, but this way the code works on windows and Linux
        __under construction!___"""
        import platform

        if not "Windows" == platform.system():
            pool = mult.Pool(processes=nProcesses)
            sim_params = [self.nSteps_between_trials, False, False, True]  # verbosity
            print("Generated pool^jobs")
            result_replica = {}
            for replica_coords, replica in self.replicas.items():
                print("Submit: ", replica_coords, replica)
                replica_result = pool.apply_async(replica.simulate, sim_params)
                result_replica.update({replica_coords: replica_result})
            print("Done Submitting")

            # Wait pool close
            pool.close()
            pool.join()

            print("Done Simulating")
            print(result_replica)
            [self.replicas.update({replica_coords: result_replica[replica_coords].get()}) for replica_coords in self.replicas]
            print("GrandFinale: ", self.replicas)
            print("Done")
        else:
            warnings.warn("Can not go parallel on windows. Not implemented! falling pack to single core.")
            self.run()

    # ATTRIBUTES:
    # getter/setters
    @property
    def exchange_information(self) -> pd.DataFrame:
        """
            contains the information about the replica exchanges
        Returns
        -------
        pd.Dataframe
            the dataframe contains the trial exchange informations.
            columns: ["nExchange", "replicaID", "replicaPositionI", "exchangeCoordinateI", "TotEI",
                     "replicaPositionJ", "exchangeCoordinateJ", "TotEJ", "doExchange"]
        """
        return self._exchange_information

    def set_simulation_steps_between_trials(self, nsteps: int) -> NoReturn:
        """
            set new nSteps between the trials.

        Parameters
        ----------
        nsteps: int
            number of steps, to be carried out inbetween the trials

        """
        self.nSteps_between_trials = nsteps
        for coord, replica in self.replicas.items():
            replica.nsteps = self.nSteps_between_trials

    # private
    def _init_exchanges(self) -> NoReturn:
        """
        initialise the exchanges
        """
        for replicaID in self.replicas:
            exchange = False
            self._exchange_information = self.exchange_information.append(
                {
                    "nExchange": self._currentTrial,
                    "replicaID": self.replicas[replicaID].replicaID,
                    "replicaPositionI": replicaID,
                    "exchangeCoordinateI": self.replicas[replicaID].exchange_parameters,
                    "TotEI": self.replicas[replicaID].calculate_total_potential_energy(),
                    "replicaPositionJ": replicaID,
                    "exchangeCoordinateJ": self.replicas[replicaID].exchange_parameters,
                    "TotEJ": self.replicas[replicaID].calculate_total_potential_energy(),
                    "doExchange": exchange,
                },
                ignore_index=True,
            )

    def _adapt_system_to_exchange_coordinate(self) -> NoReturn:
        """
        Interface function, needs to implemented. What should be done after an exchange?
        """
        raise NotImplementedError("UPS this func is not implemented please override.")
