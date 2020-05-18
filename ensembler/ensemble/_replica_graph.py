"""
.. automodule: _replica graph

"""

import numpy as np
import pandas as pd
from collections import Iterable, namedtuple
import scipy.constants as const
from typing import List, Dict, Tuple
import copy
import itertools as it

from ensembler import system, potentials as pot

class MultiReplicaApproach():
    ##Parameters - to build replica graphs
    ### EDGES - Exchange parameters - gradients
    coordinate_dimensions:int
    parameter_names:List
    exchange_dimensions:Dict
    
    ### NODES - Replicas
    nReplicas:int
    replicas:dict={}
    replica_graph_dimensions:int

    coord_set: namedtuple 

    nSteps_between_trials:int

    ##init funcs
    def _initialise_replica_graph(self, verbose:bool=False):

        coordinates = self.generate_all_node_coordinates_combinations()

        if(verbose):
            print("\nBUILD Replicas")
            print("Coordinates\n\t", coordinates,"\n")

        self.build_nodes(coordinates=coordinates)

        if(verbose):
            print("Replicas:\n\tuniqID\tcoordinates\n\t"+"\n\t".join([str(self.replicas[x].uniqueID)+"\t"+str(self.replicas[x].Nodecoordinates) for x in self.replicas]))

    ### * Node Functions
    #### Generate coordinates
    def generate_all_node_coordinates_combinations(self)->List[namedtuple]:
        self.coord_dims = list(sorted(self.exchange_dimensions))
        self.coord_names = list(sorted(self.exchange_dimensions.keys()))
        self.coord_set = namedtuple("coordinates", self.coord_names)

        #generate all parameter combinations
        if(len(self.exchange_dimensions) > 1):
            coord_it=it.product(*[list(self.exchange_dimensions[r]) for r in sorted(self.exchange_dimensions)])
            coordinates = [self.coord_set({name: x for name, x in zip(self.coord_names, x)}) for x in coord_it]
        elif(len(self.exchange_dimensions) == 1):
            coordinates = list(map(lambda x: self.coord_set(x), self.exchange_dimensions[self.coord_dims[0]]))
        else:
            raise Exception("Could not find parameters to exchange")
        
        return coordinates


    ###build nodes
    def build_nodes(self, coordinates:Dict[str, any]):
        ## deepcopy all
        self.nReplicas = len(list(coordinates))
        replicas = [copy.deepcopy(self.system) for x in range(self.nReplicas)]#generate deepcopies

        # build up graph - set parameters
        uid = 0
        for coords, replica in zip(coordinates, replicas):

            ## finalize deepcopy:
            replica.trajectory = [] #fields are not deepcopied!!!
            replica.nsteps = self.nSteps_between_trials     # set steps between trials #todo: ? Really here?
            
            ## Node properties
            setattr(replica, "uniqueID", uid)
            setattr(replica, "Nodecoordinates", coords)

            ##set parameters
            for ind, parameter_Name in enumerate(self.coord_dims):
                if (hasattr(replica, parameter_Name) or True):
                    if (isinstance(coords, Iterable)):
                        setattr(replica, parameter_Name, coords[ind])
                    else:
                        setattr(replica, parameter_Name, coords)
                else:
                    raise Exception("REPLICA INIT FAILED: Replica does not have a field: "+parameter_Name+"\n")
            replica.init_velocities() # e.g. if temperature changes

            self.replicas.update({uid: replica})
            uid+=1


class ReplicaExchange(MultiReplicaApproach):
    ##Parameters

    ##Exchange Variables
    _currentTrial:int
    nSteps_between_trials:int


    ###Exchange params/io
    exchange_information: pd.DataFrame = pd.DataFrame(columns=["nExchange", "uniqueReplicaID","replicaI", "exchangeCoordinateI", "TotEI",
                                                          "replicaJ","exchangeCoordinateJ", "TotEJ", "doExchange"])
    
    ###METROPOLIS CRITERION
    ###default Metropolis Criterion
    _defaultMetropolisCriterion = lambda self, originalParams, swappedParams: (np.greater_equal(originalParams, swappedParams) or self._defaultRandomness(originalParams, swappedParams))
    exchange_criterium = _defaultMetropolisCriterion
    exchange_offset=0

    ###random part of Metropolis Criterion:
    randomnessIncreaseFactor = 0.1
    _temperature_exchange:float= 298
    _defaultRandomness = lambda self, originalParams, swappedParams: ((1 / self.randomnessIncreaseFactor) * np.random.rand() <= np.exp(-1.0 / (const.gas_constant / 1000.0 * self._temperature_exchange) * (originalParams - swappedParams+0.0000001))) #pseudo count, if params are equal


    def __init__(self, system:system.system, exchange_dimensions:Dict[str, Iterable], exchange_criterium=None, steps_between_trials:int=10):

        #TODO do some fancy parsing
        #SET PARAMETER FIELDS

        if(isinstance(exchange_dimensions, dict)):
            self.exchange_dimensions = exchange_dimensions
        self.coordinate_dimensions = len(exchange_dimensions)  # get dimensionality
        self.parameter_names = list(self.exchange_dimensions.keys())

        #SET SYSTEM
        self.system = system

        #exchange finfo:
        self.exchange_information = pd.DataFrame(columns=["nExchange", "uniqueReplicaID","replicaI", "exchangeCoordinateI", "TotEI",
                                                          "replicaJ","exchangeCoordinateJ", "TotEJ", "doExchange"])

        if(steps_between_trials != None):
            self.nSteps_between_trials = steps_between_trials

        #initialize the replica graphs
        self.initialise()

        #exchange Criterium
        if(exchange_criterium != None):
            self.exchange_criterium = exchange_criterium


    #public functions
    def initialise(self):
        self._currentTrial = 0
        # BUILD replicas
        self._initialise_replica_graph()


    def simulate(self, ntrials:int, steps_between_trials:int=None, reset_ensemble:bool=False):
        if(reset_ensemble):
            self._currentTrial=0
            [replica.initialise(withdraw_Traj=True) for repName,replica in self.replicas.items()]
            self.exchange_information = pd.DataFrame(columns=["nExchange", "uniqueReplicaID","replicaI", "exchangeCoordinateI", "TotEI",
                                                          "replicaJ","exchangeCoordinateJ", "TotEJ", "doExchange"])

        if(isinstance(steps_between_trials, int)):
            self.set_simulation_steps_between_trials(nsteps=steps_between_trials)

        for trial in range(ntrials):
            self.run()
            self.exchange()
        self.exchange_information = self.exchange_information


    def exchange(self):
        raise NotImplementedError("This method was not implemented for "+str(__name__)+" do so!")
        pass


    def run(self):
        for replica_coords, replica in self.replicas.items():
            replica.simulate(steps=self.nSteps_between_trials)
        pass


    #getter/setters
    def get_trajectories(self)->Dict[Tuple, List]:
        return {coord:replica.trajectory for coord, replica in self.replicas.items()}


    def get_replicas_positions(self)->Dict:
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


    def get_total_energy(self)->Dict[Tuple, float]:
        [replica.updateEne() for coord, replica in self.replicas.items()]
        return {coord:replica.getTotEnergy() for coord, replica in self.replicas.items()}


    def set_simulation_steps_between_trials(self, nsteps:int):
        self.nSteps_between_trials = nsteps
        for coord, replica in self.replicas.items():
            replica.nsteps = self.nSteps_between_trials


    def set_replicas_positions(self, positions:(List or Dict)):
        """
        .. autofunction:: setReplicaPositions
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new positions.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        if(len(positions)==self.nReplicas):
            if (type(positions) == dict):
                for replicaName, position in positions.items():
                    self.replicas[replicaName].set_position(position)
            elif(isinstance(positions, Iterable)):
                for replicaName, position in zip(sorted(self.replicas), positions):
                    self.replicas[replicaName].set_position(position)
            else:
                raise Exception("Did not understand the the type of the new positions "+str(type(positions)))
        else:
            raise ValueError("Not enough positions got passed to setReplicapositions\n replicas: "+str(self.nReplicas)+"\n positions: "+str(len(positions))+"\n"+str(positions))


    def set_replicas_velocities(self, velocities:(List or Dict)):
        """
        .. autofunction:: setReplicasVelocities
            If a list is passed to this function, the replicas will be aligned by sorted and then sequentially filled up with the new velocity.
            Else a Dictionary can be passed and the directed position is transferred to the replica at the replica coordinate(key)
        :param positions:
        :return:
        """
        if(len(velocities)==self.nReplicas):
            if (type(velocities) == dict):
                for replicaName, velocity in velocities.items():
                    self.replicas[replicaName].set_velocities(velocity)
            elif(isinstance(velocities, Iterable)):
                for replicaName, velocity in zip(sorted(self.replicas), velocities):
                    self.replicas[replicaName].set_velocities(velocity)
            else:
                raise Exception("Did not understand the the type of the new positions "+str(type(velocities)))
        else:
            raise ValueError("Not enough positions got passed to setReplicapositions\n replicas: "+str(self.nReplicas)+"\n positions: "+str(len(velocities))+"\n"+str(velocities))


    def set_parameter_set(self, coordinates:List, replicas:List):
        """
            ..autofunction:: set ParameterSet
            This function is setting new coordinates to the replicas in the replica lists.
            The coordinates will be assigned sequentially in the same order to the replicas List.

        :warning: This function is Overwritting old coordinates!
        :param coordinates:
        :return:
        """

        if(self.coordinate_dimensions>1):
            self.replicas = {}
            for coords, replica in zip(coordinates, replicas):
                for ind, parameter_Name in enumerate(self.exchange_dimensions):
                    #set parameter set
                    if (hasattr(replica, parameter_Name)):
                        setattr(replica, parameter_Name, coords[ind])
                    else:
                        raise Exception("REPLICA INIT FAILDE: Replica does not have a field: "+parameter_Name+"\n")
                self.replicas.update({coords: replica})
        else:
            self.replicas = {}
            for coords, replica in zip(coordinates, replicas):
                #set parameter set
                if (hasattr(replica, self.parameter_names[0])):
                    if (isinstance(coords, Iterable)):
                        setattr(replica, self.parameter_names[0], coords[0])
                    else:
                        setattr(replica, self.parameter_names[0], coords)

                else:
                    raise Exception("REPLICA INIT FAILDE: Replica does not have a field: " + self.exchange_dimensions[0] + "\n")
                self.replicas.update({coords: replica})


    #private
    ##Exchange functions
    def _collect_replica_energies(self, verbose):
        original_totPots = self.get_total_energy()
        original_exCoord = list(sorted(original_totPots.keys()))

        replica_values = list([self.replicas[key] for key in original_exCoord])

        # SWAP exchange_coordinates params pairwise
        swapped_exCoord = self._swap_coordinates(original_exCoord)

        if (verbose):
            print("original Coords ", original_exCoord)
            print("swapped Coords ", swapped_exCoord)

        # SWAP Params to calculate energies in swapped case
        self.set_parameter_set(coordinates=swapped_exCoord, replicas=replica_values)  # swap parameters

        # Exchange coordinate paramters:
        ##scaleVel
        self._adapt_system_to_exchange_coordinate(swapped_exCoord, original_exCoord)

        ##get_swapped energies
        swapped_totPots = self.get_total_energy()  # calc swapped parameter Energies

        ##scale Vel back
        self._adapt_system_to_exchange_coordinate(swapped_exCoord, original_exCoord)
        self.set_parameter_set(coordinates=original_exCoord, replicas=replica_values)  # swap back parameters

        return original_exCoord, original_totPots, swapped_exCoord, swapped_totPots


    def _do_exchange(self, exchanges_to_make, original_exCoord, original_totPots, swapped_totPots, verbose):

        if (self.exchange_offset == 1):
            exchange = False
            partner1ID = partner2ID = original_exCoord[0]
            self.exchange_information = self.exchange_information.append(
                {"nExchange": self._currentTrial, "uniqueReplicaID": self.replicas[partner2ID].uniqueID,
                 "replicaI": partner1ID, "exchangeCoordinateI": partner2ID, "TotEI": swapped_totPots[partner2ID],
                 "replicaJ": partner2ID, "exchangeCoordinateJ": partner1ID, "TotEJ": swapped_totPots[partner1ID],
                 "doExchange": exchange}, ignore_index=True)

        for (partner1ID, partner2ID), exchange in exchanges_to_make.items():
            if (exchange):
                if (verbose): print(
                    "Exchanging: " + str(partner1ID) + "\t" + str(partner2ID) + "\t" + str(exchange) + "\n")
                partner1 = self.replicas[partner1ID]
                partner2 = self.replicas[partner2ID]

                # T=self.parameter_names[0]
                exchange_param = "trajectory"
                param = getattr(partner1, self.exchange_param)
                setattr(partner1, self.exchange_param, getattr(partner2, self.exchange_param))
                setattr(partner2, self.exchange_param, param)
                tmp_id = getattr(partner1, "uniqueID")
                setattr(partner1, "uniqueID", getattr(partner2, "uniqueID"))
                setattr(partner2, "uniqueID", tmp_id)
            else:
                if (verbose): print("not Exchanging: " + str(partner1ID) + " / " + str(partner2ID) + " \n")
            # add exchange info line here!

            self.exchange_information = self.exchange_information.append(
                {"nExchange": self._currentTrial, "uniqueReplicaID": self.replicas[partner1ID].uniqueID,
                 "replicaI": partner1ID, "exchangeCoordinateI": partner1ID, "TotEI": original_totPots[partner1ID],
                 "replicaJ": partner2ID, "exchangeCoordinateJ": partner2ID, "TotEJ": original_totPots[partner2ID],
                 "doExchange": exchange}, ignore_index=True)
            self.exchange_information = self.exchange_information.append(
                {"nExchange": self._currentTrial, "uniqueReplicaID": self.replicas[partner2ID].uniqueID,
                 "replicaI": partner2ID, "exchangeCoordinateI": partner2ID, "TotEI": swapped_totPots[partner1ID],
                 "replicaJ": partner1ID, "exchangeCoordinateJ": partner1ID, "TotEJ": swapped_totPots[partner2ID],
                 "doExchange": exchange}, ignore_index=True)

        if (self.exchange_offset == 0):
            exchange = False
            partner1ID = partner2ID = original_exCoord[-1]
            self.exchange_information = self.exchange_information.append(
                {"nExchange": self._currentTrial, "uniqueReplicaID": self.replicas[partner2ID].uniqueID,
                 "replicaI": partner1ID, "exchangeCoordinateI": partner2ID, "TotEI": swapped_totPots[partner2ID],
                 "replicaJ": partner2ID, "exchangeCoordinateJ": partner1ID, "TotEJ": swapped_totPots[partner1ID],
                 "doExchange": exchange}, ignore_index=True)


    def _swap_coordinates(self):
        raise NotImplementedError("UPS this func is not implemented please override.")


    def _adapt_system_to_exchange_coordinate(self, swapped_exCoord, original_exCoord):
        raise NotImplementedError("UPS this func is not implemented please override.")
