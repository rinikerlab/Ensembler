"""
.. automodule: replica_approach_dynamic_parameters
    This module shall be used to implement subclasses of ensemble.
    It is a class, that is using multiple system. It can be used for RE or Conveyor belt
"""
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from ensembler import potentials as pot
from ensembler.ensemble._replica_graph import ReplicaExchange
from ensembler.integrator import stochastic
from ensembler.system import perturbed_system


class ConveyorBelt(ReplicaExchange):
    '''
        Conveyor belt ensemble class
        organizes the replicas and their coupling
    '''

    _parameter_name: str = "lam"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1
    exchange_dimensions: Dict[str, np.array]
    nSteps_between_trials: int = 2

    exchange_information: pd.DataFrame = pd.DataFrame(columns=["Step", "capital_lambda", "TotE", "biasE", "doAccept"])
    system_trajs: dict = {}

    def __str__(self):
        '''
        :return: ensemble state string
        '''
        outstr = ''
        print(self.replicas)
        for i in self.replicas:
            outstr += '{:.1f}{:10.2f}{:10.3f}\n'.format(i, self.replicas[i]._currentLam,
                                                        float(self.replicas[i].getTotEnergy()))
        return outstr

    def __repr__(self):
        '''
        :return: ensemble state string
        '''
        return self.__str__()

    def __init__(self, capital_lambda: float, nReplicas: int,
                 system=perturbed_system.perturbedSystem(temperature=300.0, lam=0.0,
                                                         potential=pot.OneD.linearCoupledPotentials(),
                                                         integrator=stochastic.metropolisMonteCarloIntegrator()),
                 build=False):
        '''
        initialize Ensemble object
        :param capital_lambda: state of ensemble, 0 <= capital_lambda < pi
        :param num: number of replicas
        :param system: a system1D instance
        :param build: build memory?whconda
        
        '''
        assert 0.0 <= capital_lambda <= 2 * np.pi, "capital_lambda not allowed"
        assert nReplicas >= 1, "At least one system is needed"

        self.nReplicas = nReplicas
        self.system = system
        self.capital_lambda = capital_lambda
        self.build = build  # build

        self.dis = 2.0 * np.pi / nReplicas
        self.exchange_dimensions = {self._parameter_name: np.arange(0, 2 * np.pi, self.dis)}

        self._temperature_exchange = system.temperature

        self.initialise()

        self.exchange_information: pd.DataFrame = pd.DataFrame(
            columns=["Step", "capital_lambda", "TotE", "biasE", "doAccept"])
        self.system_trajs: dict = {}

    # public functions
    def initialise(self):
        ##Simulation
        self._currentTrial = 0
        self.reject = 0

        # initialize memory variables
        self.num_gp = None
        self.mem_fc = None
        self.mem = None
        self.gp_spacing = None
        self.biasene = None
        self.init_mem()

        # BUILD replicas
        self._initialise_replica_graph()

        ## * Conveyor belt specifics
        for i in self.replicas:
            self.replicas[i]._update_dHdlambda()

    def simulate(self, ntrials: int, steps_between_trials: int = None, reset_ensemble: bool = False,
                 verbosity: bool = True):

        if (isinstance(steps_between_trials, int)):
            self.set_simulation_steps_between_trials(nsteps=steps_between_trials)

        for trial in tqdm(range(ntrials), desc="Trials: ", mininterval=1.0, leave=verbosity):
            self.run()
            self.accept_move()

        # self.exchange_information = self.exchange_information

    def run(self):
        self._currentTrial += 1
        for replica_coords, replica in self.replicas.items():
            replica.simulate(steps=self.nSteps_between_trials, verbosity=False)

    def accept_move(self):
        self.state = []

        # metropolis criterium for moving capital_lambda?
        oldEne = self.calculate_conveyorBelt_totEne()
        oldBiasene = self.biasene
        oldBlam = self.capital_lambda

        self.capital_lambda += (np.random.rand() * 2.0 - 1.0) * np.pi / 4.0
        self.capital_lambda = self.capital_lambda % (2.0 * np.pi)
        self.updateBlam(self.capital_lambda)

        newEne = self.calculate_conveyorBelt_totEne()
        if self._defaultMetropolisCriterion(originalParams=oldEne, swappedParams=newEne):
            for i in self.replicas:
                self.replicas[i]._update_dHdlambda()

            self.exchange_information = self.exchange_information.append(
                {"Step": self._currentTrial, "capital_lambda": self.capital_lambda, "TotE": float(newEne),
                 "biasE": float(newEne), "doAccept": True}, ignore_index=True)

        else:
            self.reject += 1
            self.updateBlam(oldBlam)

            for i in self.replicas:
                self.replicas[i]._update_dHdlambda()

            self.exchange_information = self.exchange_information.append(
                {"Step": self._currentTrial, "capital_lambda": oldBlam, "TotE": float(oldEne),
                 "biasE": float(oldBiasene), "doAccept": False}, ignore_index=True)

        if self.build:
            self.build_mem()

    def revert(self):
        '''
        reverts last propagation step
        :return: None
        '''
        for j in self.replicas:
            self.replicas[j].revert()
        self.calculate_conveyorBelt_totEne()
        self.exchange_information = self.exchange_information[:-1]

    def add_replica(self, clam: float, addNReplicas: int = 1) -> None:
        '''
            Not Implemented!!!
        adds a replica to the ensemble
        :return: None
        '''
        raise NotImplementedError("Please Implement this function!")

    # PRIVATE functions
    ## * Move the belt
    def calculate_conveyorBelt_totEne(self) -> float:
        '''
        calculates energy of Conveyor Belt Ensemble
        :return: total energy of the Conveyor Belt Ensemble.
        :rtype: float
        '''
        ene = 0.0
        for i in self.replicas:
            ene += self.replicas[i]._currentTotPot
            ene += self.replicas[i]._currentTotKin if (not np.isnan(self.replicas[i]._currentTotKin)) else 0
        ene = ene + self.biasene
        return ene

    def calc_lam(self, capital_lambda: float, i: int) -> float:
        '''
        calculates lam_i for replica i depending on ensemble state capital_lambda

        :param capital_lambda: state of ensemble 0 <= capital_lambda < 2 pi
        :type capital_lambda: float
        :param i: index of replica
        :type i: int
        :return: lambda of replica i
        :rtype: float
        '''

        ome = (capital_lambda + i * self.dis) % (2. * np.pi)
        if ome > np.pi:
            ome = 2.0 * np.pi - ome
        return ome / np.pi

    def updateBlam(self, capital_lambda: float) -> float:
        '''
        updates the state of the ensemble and the replicas accordingly
        :param capital_lambda: capital lambda 0 <= capital_lambda < 2 pi
        :type capital_lambda: float
        :return: capital_lambda
        :rtype: float
        '''
        self.capital_lambda = capital_lambda
        for i in self.replicas:
            self.replicas[i].set_lam(self.calc_lam(capital_lambda, i))
        self.apply_mem()

        return capital_lambda

    ## * Bias Memory Functions
    def init_mem(self):
        '''
        initializes memory
        :return: None
        '''
        self.num_gp = 11
        self.mem_fc = 0.0001
        #        self.mem=np.array([2.2991 ,  2.00274,  1.84395,  1.83953,  2.0147])
        #        memory for perturbed hosc, alpha=10.0, gamma=0.0, 8 replica, num_gp=6, fc=0.00001, 1E6 steps
        self.mem = np.zeros(self.num_gp - 1)
        self.gp_spacing = self.dis / float(self.num_gp - 1.0)
        self.biasene = 0.0
        # print('Distance: ', self.dis, self.dis / np.pi)
        # print('GP Distance: ', self.gp_spacing, self.gp_spacing / np.pi)
        # print('Gridpoints: ', np.linspace(0, self.num_gp - 1, self.num_gp) * self.gp_spacing)
        # print('Gridpoints: ', np.linspace(0, self.num_gp - 1, self.num_gp) * self.gp_spacing / np.pi)

    def build_mem(self):
        '''
        increments biasing memory
        :return: None
        '''
        active_gp = int(np.floor((self.capital_lambda % self.dis) / self.gp_spacing + 0.5))
        self.mem[active_gp % (self.num_gp - 1)] += self.mem_fc

    def apply_mem(self):
        """
            applies memory biasing
        
        :return: None
        """
        active_gp = int(np.floor((self.capital_lambda % self.dis) / self.gp_spacing + 0.5))
        dg = (self.capital_lambda % self.dis) / self.gp_spacing - float(active_gp)
        if dg < 0:
            self.biasene = self.mem[(active_gp - 1) % (self.num_gp - 1)] * self.spline(1.0 + dg) + self.mem[
                active_gp % (self.num_gp - 1)] * self.spline(-dg)
        else:
            self.biasene = self.mem[active_gp % (self.num_gp - 1)] * self.spline(dg) + self.mem[
                (active_gp + 1) % (self.num_gp - 1)] * self.spline(1.0 - dg)
        # print("{:5.2f}{:5.2f}{:8.3f}{:3d}{:8.3f}{:8.3f}{:8.3f} {:s}".format(self.capital_lambda, (self.capital_lambda%self.dis),
        # (self.capital_lambda%self.dis)/self.gp_spacing, active_gp,
        # self.gp_spacing*active_gp, dg, ene, np.array2string(self.mem)))

    ## * Trajectories 
    def get_trajs(self) -> (pd.DataFrame, Dict[int, pd.DataFrame]):
        """
        returns all Trajectories of this Ensemble.
        
        :return: Conveyor Belt Trajectory, Replica Trajectories.
        :rtype: (pd.DataFrame, Dict[int, pd.DataFrame])
        """
        return self.get_conveyorbelt_trajectory(), self.get_replica_trajectories()

    def get_conveyorbelt_trajectory(self) -> pd.DataFrame:
        """
        get_conveyorbelt_trajectory returns the pandas DataFrame of the conveyorbelt trajectory
        
        :return: conveyorbelt_trajectory
        :rtype: pd.DataFrame
        """
        return self.exchange_information

    def get_replica_trajectories(self) -> Dict[int, pd.DataFrame]:
        """
        get_replica_trajectories 
        
        :return: trajectories of all replicas
        :rtype: Dict[int, pd.DataFrame]
        """
        self.system_trajs = {}
        for i in self.replicas:
            self.system_trajs.update({i: self.replicas[i].getTrajectory()})
        return self.system_trajs

    def clear_all_trajs(self):
        '''
        deletes trajectories of replicas
        :return: None
        '''
        self.system_trajs = {}
        self.exchange_information = []

    # Todo: should be inherited.
    def set_simulation_steps_between_trials(self, nsteps: int):
        self.nSteps_between_trials = nsteps
        for coord, replica in self.replicas.items():
            replica.nsteps = self.nSteps_between_trials

    # Todo: PUT SOMEWHERE ELSE OR NUMPY?.
    @staticmethod
    def spline(dg):
        '''
        calculates the value of the spline function depending on the deviation dg from the grid point
        :param dg: deviation from gridpoint (absolute value)
        :return: value of spline (float)
        '''
        if dg < 0.0:
            print('distance smaller than 0')
        elif dg < 1.0:
            return 1.0 - 3.0 * dg * dg + 2 * dg * dg * dg
        else:
            return 0.0
