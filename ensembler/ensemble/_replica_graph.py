"""
.. automodule: _replica graph

"""
import copy
import itertools as it
import multiprocessing as mult
from collections import namedtuple
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import scipy.constants as const
from tqdm import tqdm_notebook as tqdm

from ensembler.ensemble import exchange_pattern
from ensembler.util.basic_class import super_baseClass
from ensembler.util.ensemblerTypes import system


class MultiReplicaApproach(super_baseClass):
    ##Parameters - to build replica graphs
    ### EDGES - Exchange parameters - gradients
    coordinate_dimensions: int
    parameter_names: List
    exchange_dimensions: Dict

    ### NODES - Replicas
    nReplicas: int
    replicas: dict = {}
    replica_graph_dimensions: int

    coord_set: namedtuple

    nSteps_between_trials: int

    ##init funcs
    def _initialise_replica_graph(self, verbose: bool = False):

        coordinates = self.generate_all_node_coordinates_combinations()

        if (verbose):
            print("\nBUILD Replicas")
            print("Coordinates\n\t", coordinates, "\n")

        self.build_nodes(coordinates=coordinates)

        if (verbose):
            print("Replicas:\n\treplicaID\tcoordinates\n\t" + "\n\t".join(
                [str(self.replicas[x].uniqueID) + "\t" + str(self.replicas[x].Nodecoordinates) for x in self.replicas]))

    ### * Node Functions
    #### Generate coordinates
    def generate_all_node_coordinates_combinations(self) -> List[namedtuple]:
        self.coord_dims = list(sorted(self.exchange_dimensions))
        self.coord_names = list(sorted(self.exchange_dimensions.keys()))
        self.coord_set = namedtuple("coordinates", self.coord_names)

        # make pickleable
        import __main__
        setattr(__main__, self.coord_set.__name__, self.coord_set)
        self.coord_set.__module__ = "__main__"

        # generate all parameter combinations
        if (len(self.exchange_dimensions) > 1):
            coord_it = list(it.product(*[list(self.exchange_dimensions[r]) for r in sorted(self.exchange_dimensions)]))
            coordinates = [self.coord_set(**{name: x for name, x in zip(self.coord_names, x)}) for x in coord_it]
        elif (len(self.exchange_dimensions) == 1):
            coordinates = list(map(lambda x: self.coord_set(x), self.exchange_dimensions[self.coord_dims[0]]))
        else:
            raise Exception("Could not find parameters to exchange")

        return coordinates

    ###build nodes
    def build_nodes(self, coordinates: Dict[str, any]):
        ## deepcopy all
        self.nReplicas = len(list(coordinates))
        replicas = [copy.deepcopy(self.system) for x in range(self.nReplicas)]  # generate deepcopies

        # build up graph - set parameters
        replicaID = 0
        for coords, replica in zip(coordinates, replicas):

            ## finalize deepcopy:
            # replica.trajectory = [] #fields are not deepcopied!!!
            replica.nsteps = self.nSteps_between_trials  # set steps between trials #todo: ? Really here?

            ## Node properties
            setattr(replica, "replicaID", replicaID)
            setattr(replica, "Nodecoordinates", coords)

            ##set parameters
            parameter_set = {}
            for ind, parameter_Name in enumerate(self.coord_dims):
                if (hasattr(replica, parameter_Name) or True):
                    if (isinstance(coords, Iterable)):
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


class ReplicaExchange(MultiReplicaApproach):
    ##Parameters
    verbose: bool = False
    ##Exchange Variables
    _currentTrial: int
    nSteps_between_trials: int
    _exchange_pattern: exchange_pattern.Exchange_pattern = None

    ###Exchange params/io
    exchange_information: pd.DataFrame = pd.DataFrame(
        columns=["nExchange", "replicaID", "replicaPositionI", "exchangeCoordinateI", "TotEI",
                 "replicaPositionJ", "exchangeCoordinateJ", "TotEJ", "doExchange"])

    ###METROPOLIS CRITERION
    ###default Metropolis Criterion
    _defaultMetropolisCriterion = lambda self, originalParams, swappedParams: (
            np.greater_equal(originalParams, swappedParams) or self._defaultRandomness(originalParams,
                                                                                       swappedParams))
    exchange_criterium = _defaultMetropolisCriterion

    ###random part of Metropolis Criterion:
    randomnessIncreaseFactor = 0.1
    _temperature_exchange: float = 298
    _defaultRandomness = lambda self, originalParams, swappedParams: (
            (1 / self.randomnessIncreaseFactor) * np.random.rand() <= np.exp(
        -1.0 / (const.gas_constant / 1000.0 * self._temperature_exchange) * (
                originalParams - swappedParams + 0.0000001)))  # pseudo count, if params are equal

    def __init__(self, system: system, exchange_dimensions: Dict[str, Iterable], exchange_criterium=None,
                 steps_between_trials: int = 10):

        # TODO do some fancy parsing
        # SET PARAMETER FIELDS

        if (isinstance(exchange_dimensions, dict)):
            self.exchange_dimensions = exchange_dimensions
        self.coordinate_dimensions = len(exchange_dimensions)  # get dimensionality
        self.parameter_names = list(self.exchange_dimensions.keys())

        # SET SYSTEM
        self.system = system

        # exchange finfo:
        self.exchange_information = pd.DataFrame(
            columns=["nExchange", "uniqueReplicaID", "replicaI", "exchangeCoordinateI", "TotEI",
                     "replicaJ", "exchangeCoordinateJ", "TotEJ", "doExchange"])

        if (steps_between_trials != None):
            self.nSteps_between_trials = steps_between_trials

        # initialize the replica graphs
        self.initialise()

        # exchange Criterium
        if (exchange_criterium != None):
            self.exchange_criterium = exchange_criterium

        if (isinstance(self._exchange_pattern, type(None))):
            self._exchange_pattern = exchange_pattern.localExchangeScheme(replica_graph=self)

    # public functions
    def initialise(self):
        self._currentTrial = 0
        # BUILD replicas
        self._initialise_replica_graph()
        self._init_exchanges()

    def simulate(self, ntrials: int, steps_between_trials: int = None, reset_ensemble: bool = False):
        if (reset_ensemble):
            self._currentTrial = 0
            [replica.initialise(withdraw_Traj=True) for repName, replica in self.replicas.items()]
            self.exchange_information = pd.DataFrame(
                columns=["nExchange", "replicaID", "replicaPositionI", "exchangeCoordinateI", "TotEI",
                         "replicaPositionJ", "exchangeCoordinateJ", "TotEJ", "doExchange"])
            self._init_exchanges()

        if (isinstance(steps_between_trials, int)):
            self.set_simulation_steps_between_trials(nsteps=steps_between_trials)

        for _ in tqdm(range(ntrials), desc="Running trials", leave=True):
            self.run()
            self.exchange()

    def exchange(self):
        self._exchange_pattern.exchange(verbose=self.verbose)

    def run(self, verbosity: bool = False):
        for replica_coords, replica in self.replicas.items():
            replica.simulate(steps=self.nSteps_between_trials, withdrawTraj=False, initSystem=False,
                             verbosity=verbosity)

    def _run_parallel(self, verbosity: bool = False, nProcesses: int = 4):
        """this is an ugly work around, but this way the code works on windows and Linux
        __under construction!___"""
        pool = mult.Pool(processes=nProcesses)
        sim_params = [self.nSteps_between_trials, False, False, True]  # verbosity
        print("Generated pool^jobs")
        result_replica = {}
        for replica_coords, replica in self.replicas.items():
            print("Submit: ", replica_coords, replica)
            replica_result = pool.apply_async(replica.simulate, sim_params)
            result_replica.update({replica_coords: replica_result})
        print("Done Submitting")

        #Wait pool close
        pool.close()
        pool.join()

        print("Done Simulating")
        print(result_replica)
        [self.replicas.update({replica_coords: result_replica[replica_coords].get()}) for replica_coords in
         self.replicas]
        print("GrandFinale: ", self.replicas)
        print("Done")


    # getter/setters
    def get_trajectories(self) -> Dict[Tuple, List]:
        return {coord: replica.getTrajectory() for coord, replica in self.replicas.items()}

    def get_replicas_positions(self) -> Dict:
        """
        .. autofunction:: getReplicaPositions
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new positions.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        vals_dict = {}
        for replicaName, replica in self.replicas.items():
            vals_dict.update({replicaName: getattr(replica, "_currentPosition")})
        return vals_dict

    def get_replicas_velocities(self) -> Dict:
        """
        .. autofunction:: getReplicaPositions
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new positions.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        vals_dict = {}
        for replicaName, replica in self.replicas.items():
            vals_dict.update({replicaName: getattr(replica, "_currentVelocities")})
        return vals_dict

    def get_replicas_current_states(self) -> Dict:
        """
        .. autofunction:: getReplicaCurrentStates
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new positions.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        vals_dict = {}
        for replicaName, replica in self.replicas.items():
            vals_dict.update({replicaName: replica.getCurrentState()})
        return vals_dict

    def get_total_energy(self) -> Dict[Tuple, float]:
        [replica._updateEne() for coord, replica in self.replicas.items()]
        return {coord: replica.getTotEnergy() for coord, replica in self.replicas.items()}

    def set_simulation_steps_between_trials(self, nsteps: int):
        self.nSteps_between_trials = nsteps
        for coord, replica in self.replicas.items():
            replica.nsteps = self.nSteps_between_trials

    def set_replicas_positions(self, positions: (List or Dict)):
        """
        .. autofunction:: setReplicaPositions
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new positions.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        if (len(positions) == self.nReplicas):
            if (type(positions) == dict):
                for replicaName, position in positions.items():
                    self.replicas[replicaName].set_position(position)
            elif (isinstance(positions, Iterable)):
                for replicaName, position in zip(sorted(self.replicas), positions):
                    self.replicas[replicaName].set_position(position)
            else:
                raise Exception("Did not understand the the type of the new positions " + str(type(positions)))
        else:
            raise ValueError("Not enough positions got passed to setReplicapositions\n replicas: " + str(
                self.nReplicas) + "\n positions: " + str(len(positions)) + "\n" + str(positions))

    def set_replicas_velocities(self, velocities: (List or Dict)):
        """
        .. autofunction:: setReplicasVelocities
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new velocity.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        if (len(velocities) == self.nReplicas):
            if (type(velocities) == dict):
                for replicaName, velocity in velocities.items():
                    self.replicas[replicaName].set_velocities(velocity)
            elif (isinstance(velocities, Iterable)):
                for replicaName, velocity in zip(sorted(self.replicas), velocities):
                    self.replicas[replicaName].set_velocities(velocity)
            else:
                raise Exception("Did not understand the the type of the new positions " + str(type(velocities)))
        else:
            raise ValueError("Not enough positions got passed to setReplicapositions\n replicas: " + str(
                self.nReplicas) + "\n positions: " + str(len(velocities)) + "\n" + str(velocities))

    def set_parameter_set(self, coordinates: List, replicas: List):
        """
            ..autofunction:: set ParameterSet
            This function is setting new coordinates to the replicas in the replica lists.
            The coordinates will be assigned sequentially in the same order to the replicas List.

        :warning: This function is Overwritting old coordinates!
        :param coordinates:
        :return:
        """

        if (self.coordinate_dimensions > 1):
            self.replicas = {}
            for coords, replica in zip(coordinates, replicas):
                for ind, parameter_Name in enumerate(self.exchange_dimensions):
                    parameters = list(self.exchange_dimensions[self.parameter_Name])

                    # set parameter set
                    if (hasattr(replica, parameter_Name)):
                        setattr(replica, parameter_Name, parameters[coords[ind]])
                    else:
                        raise Exception("REPLICA INIT FAILDE: Replica does not have a field: " + parameter_Name + "\n")
                self.replicas.update({coords: replica})
        else:
            self.replicas = {}
            parameters = list(self.exchange_dimensions[self.parameter_names[0]])

            for coords, replica in zip(coordinates, replicas):
                # set parameter set
                # print(coords, replica)
                if (hasattr(replica, self.parameter_names[0])):
                    setattr(replica, self.parameter_names[0], parameters[coords])

                else:
                    raise Exception(
                        "REPLICA INIT FAILDE: Replica does not have a field: " + self.exchange_dimensions[0] + "\n")
                self.replicas.update({coords: replica})

    # private
    def _init_exchanges(self):
        for replicaID in self.replicas:
            exchange = False
            self.exchange_information = self.exchange_information.append(
                {"nExchange": self._currentTrial, "replicaID": self.replicas[replicaID].replicaID,
                 "replicaPositionI": replicaID, "exchangeCoordinateI": self.replicas[replicaID].exchange_parameters,
                 "TotEI": self.replicas[replicaID].totPot(),
                 "replicaPositionJ": replicaID, "exchangeCoordinateJ": self.replicas[replicaID].exchange_parameters,
                 "TotEJ": self.replicas[replicaID].totPot(),
                 "doExchange": exchange}, ignore_index=True)

    def _adapt_system_to_exchange_coordinate(self, swapped_exCoord, original_exCoord):
        raise NotImplementedError("UPS this func is not implemented please override.")
