"""
.. automodule: replica_approach_dynamic_parameters
    This module shall be used to implement subclasses of ensemble.
    It is a class, that is using multiple system. It can be used for RE or Conveyor belt
"""
import numpy as np
import pandas as pd
from scipy import constants as const
from tqdm.notebook import tqdm

from ensembler import potentials as pot
from ensembler.ensemble._replica_graph import _mutliReplicaApproach
from ensembler.samplers import stochastic
from ensembler.system import perturbed_system
from ensembler.util.ensemblerTypes import systemCls, Dict, Tuple, NoReturn


class conveyorBelt(_mutliReplicaApproach):
    """
        Conveyor belt ensemble class
    organizes the replicas and their coupling
    """

    _parameter_name: str = "lam"
    coordinate_dimensions: int = 1
    replica_graph_dimensions: int = 1
    exchange_dimensions: Dict[str, np.array]
    nSteps_between_trials: int = 1

    exchange_information: pd.DataFrame = pd.DataFrame(columns=["Step", "capital_lambda", "TotE", "biasE", "doAccept"])
    system_trajs: dict = {}

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

    def __str__(self):
        outstr = "{:<5s}{:<10s}{:<10s}\n".format("i", "lambda_i", "E_i")
        outstr += "-" * 25 + "\n"
        for i in self.replicas:
            outstr += "{:5d}{:10.2f}{:10.3f}\n".format(i, self.replicas[i].lam, float(self.replicas[i].total_system_energy))
        return outstr

    def __repr__(self):
        return self.__str__()

    def __init__(
        self,
        capital_lambda: float,
        n_replicas: int,
        system: systemCls = perturbed_system.perturbedSystem(
            temperature=300.0,
            lam=0.0,
            potential=pot.OneD.linearCoupledPotentials(
                Va=pot.OneD.harmonicOscillatorPotential(k=1, x_shift=0), Vb=pot.OneD.harmonicOscillatorPotential(k=2, x_shift=0)
            ),
            sampler=stochastic.metropolisMonteCarloIntegrator(),
        ),
        build: bool = False,
    ):
        """
            initialize Ensemble object

        Parameters
        ----------
        capital_lambda: float
            state of ensemble, 0 <= capital_lambda < pi
        n_replicas: int
            number of replicas
        system: systemCls, optional
            a system1D instance
        build:bool, optional
            build memory?
        """

        assert 0.0 <= capital_lambda <= 2 * np.pi, "capital_lambda not allowed"
        assert n_replicas >= 1, "At least one system is needed"
        super().__init__()

        self.system = system
        self.capital_lambda = capital_lambda
        self.build = build  # build

        self.dis = 2.0 * np.pi / n_replicas
        self.exchange_dimensions = {
            self._parameter_name: [self.calculate_replica_lambda(self.capital_lambda, i) for i in range(n_replicas)]
        }

        self._temperature_exchange = system.temperature

        self.initialise()

        self.system_trajs: dict = {}

    # public functions
    def initialise(self) -> NoReturn:
        """
            Initialises a conveyor belt ensemble: deletes biasing potential,
            initialises the replica graph and updates the systems.

        Returns
        -------
        NoReturn

        """

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
            self.replicas[i].lam = self.calculate_replica_lambda(self.capital_lambda, i)
            self.replicas[i].update_current_state()
            self.replicas[i].clear_trajectory()
        self.exchange_information = pd.DataFrame(
            [
                {
                    "Step": self._currentTrial,
                    "capital_lambda": self.capital_lambda,
                    "TotE": self.calculate_total_ensemble_energy(),
                    "biasE": self.biasene,
                    "doAccept": True,
                }
            ]
        )

    def simulate(self, ntrials: int, nSteps_between_trials: int = 1, reset_ensemble: bool = False, verbosity: bool = True):
        """
            Integrates the conveyor belt ensemble

        Parameters
        ----------
        ntrials:int
            Number of conveyor belt steps
        nSteps_between_trials: int, optional
            number of integration steps of replicas between a move of the conveyor belt  (Default: None)
        reset_ensemble: bool, optional
            reset ensemble for starting the simulation? (Default: False)
        verbosity: bool, optional
            verbose output? (Default: False)

        Returns
        -------
        NoReturn

        """

        if isinstance(nSteps_between_trials, int):
            self.set_simulation_n_steps_between_trials(n_steps=nSteps_between_trials)

        self.__tmp_exchange_traj = []
        for _ in tqdm(range(ntrials), desc="Trials: ", mininterval=1.0, leave=verbosity):
            self.accept_move()
            self.run()

        self.exchange_information = pd.concat([self.exchange_information, pd.DataFrame(self.__tmp_exchange_traj)], ignore_index=True)

        # self.exchange_information = self.exchange_information

    def run(self, verbosity: bool = False) -> NoReturn:
        """
        Integrates the systems of the ensemble for the :var:nSteps_between_trials.
        """

        self._currentTrial += 1
        for replica_coords, replica in self.replicas.items():
            replica.simulate(steps=self.nSteps_between_trials, verbosity=verbosity)

    def accept_move(self) -> NoReturn:
        """
        Performs one trial move of the capital lambda, either accepts or rejects it and
        updates the lambdas of all replicas.
        """

        self.state = []

        # metropolis criterium for moving capital_lambda?
        oldEne = self.calculate_total_ensemble_energy()
        oldBiasene = self.biasene
        oldBlam = self.capital_lambda

        self.capital_lambda += (np.random.rand() * 2.0 - 1.0) * np.pi / 4.0
        self.capital_lambda = self.capital_lambda % (2.0 * np.pi)
        self.update_all_lambda(self.capital_lambda)

        newEne = self.calculate_total_ensemble_energy()
        if self._default_metropolis_criterion(originalParams=oldEne, swappedParams=newEne):
            for i in self.replicas:
                self.replicas[i]._update_dHdLambda()

            self.__tmp_exchange_traj.append(
                {
                    "Step": self._currentTrial,
                    "capital_lambda": self.capital_lambda,
                    "TotE": float(newEne),
                    "biasE": self.biasene,
                    "doAccept": True,
                }
            )
        else:
            self.reject += 1
            self.update_all_lambda(oldBlam)

            for i in self.replicas:
                self.replicas[i]._update_dHdLambda()

            self.__tmp_exchange_traj.append(
                {
                    "Step": self._currentTrial,
                    "capital_lambda": oldBlam,
                    "TotE": float(oldEne),
                    "biasE": float(oldBiasene),
                    "doAccept": False,
                }
            )

        if self.build:
            self.build_mem()

    def revert(self) -> NoReturn:
        """
        reverts last propagation step

        """
        for j in self.replicas:
            self.replicas[j].revert()
        self.calculate_total_ensemble_energy()
        self.exchange_information = self.exchange_information[:-1]

    def add_replica(self, clam: float, add_n_replicas: int = 1) -> NoReturn:
        """
            Not Implemented!!!
        adds a replica to the ensemble
        """
        raise NotImplementedError("Please Implement this function!")

    # PRIVATE functions
    ## * Move the belt
    def calculate_total_ensemble_energy(self) -> float:
        """
            calculates energy of Conveyor Belt Ensemble

        Returns
        -------
        float
            total energy of the Conveyor Belt Ensemble.
        """

        ene = 0.0
        for i in self.replicas:
            ene += self.replicas[i]._currentTotPot
            ene += self.replicas[i]._currentTotKin if (not np.isnan(self.replicas[i]._currentTotKin)) else 0
        ene = ene + self.biasene
        return ene

    def calculate_replica_lambda(self, capital_lambda: float, i: int) -> float:
        """

        Parameters
        ----------
        capital_lambda: float
            state of ensemble 0 <= capital_lambda < 2 pi
        i: int
            index of replica

        Returns
        -------
        float
            lambda of replica i
        """

        ome = (capital_lambda + i * self.dis) % (2.0 * np.pi)
        if ome > np.pi:
            ome = 2.0 * np.pi - ome
        return ome / np.pi

    def update_all_lambda(self, capital_lambda: float) -> float:
        """
            updates the state of the ensemble and the replicas accordingly

        Parameters
        ----------
        capital_lambda:float
            capital lambda 0 <= capital_lambda < 2 pi
        Returns
        -------
        float
            capital_lambda
        """
        """
        :param capital_lambda: 
        :type capital_lambda: float
        :return: capital_lambda
        :rtype: float
        """
        self.capital_lambda = capital_lambda
        for i in self.replicas:
            self.replicas[i].lam = self.calculate_replica_lambda(capital_lambda, i)
        self.apply_mem()

        return capital_lambda

    ## * Bias Memory Functions
    def init_mem(self) -> NoReturn:
        """
        initializes memory
        """
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

    def build_mem(self) -> NoReturn:
        """
        increments biasing memory
        """
        active_gp = int(np.floor((self.capital_lambda % self.dis) / self.gp_spacing + 0.5))
        self.mem[active_gp % (self.num_gp - 1)] += self.mem_fc

    def apply_mem(self) -> NoReturn:
        """
        applies memory biasing
        """

        active_gp = int(np.floor((self.capital_lambda % self.dis) / self.gp_spacing + 0.5))
        dg = (self.capital_lambda % self.dis) / self.gp_spacing - float(active_gp)
        if dg < 0:
            self.biasene = self.mem[(active_gp - 1) % (self.num_gp - 1)] * self.spline(1.0 + dg) + self.mem[
                active_gp % (self.num_gp - 1)
            ] * self.spline(-dg)
        else:
            self.biasene = self.mem[active_gp % (self.num_gp - 1)] * self.spline(dg) + self.mem[
                (active_gp + 1) % (self.num_gp - 1)
            ] * self.spline(1.0 - dg)
        # print("{:5.2f}{:5.2f}{:8.3f}{:3d}{:8.3f}{:8.3f}{:8.3f} {:s}".format(self.capital_lambda, (self.capital_lambda%self.dis),
        # (self.capital_lambda%self.dis)/self.gp_spacing, active_gp,
        # self.gp_spacing*active_gp, dg, ene, np.array2string(self.mem)))

    ## * Trajectories
    def get_trajs(self) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
            returns all Trajectories of this Ensemble.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]
            Conveyor Belt Trajectory, Replica Trajectories.
        """
        return self.get_conveyorbelt_trajectory(), self.get_replica_trajectories()

    def get_conveyorbelt_trajectory(self) -> pd.DataFrame:
        """
        get_conveyorbelt_trajectory returns the pandas DataFrame of the conveyorbelt trajectory

        Returns
        -------
        pd.DataFrame
            conveyorbelt_trajectory
        """

        return self.exchange_information

    def get_replica_trajectories(self) -> Dict[int, pd.DataFrame]:
        """
                get_replica_trajectories
        Returns
        -------
        Dict[int, pd.DataFrame]
            trajectories of all replicas
        """

        self.system_trajs = {}
        for i in self.replicas:
            self.system_trajs.update({i: self.replicas[i].trajectory})
        return self.system_trajs

    def clear_all_trajs(self) -> NoReturn:
        """
        deletes trajectories of replicas

        """
        self.system_trajs = {}
        for i in self.replicas:
            self.replicas[i].clear_trajectory()
        self.exchange_information = pd.DataFrame(columns=["Step", "capital_lambda", "TotE", "biasE", "doAccept"])

    def set_simulation_n_steps_between_trials(self, n_steps: int) -> NoReturn:
        """
                    Sets the integration steps of the replicas between a trail move.

        Parameters
        ----------
        n_steps:int
            number of steps
        """
        self.nSteps_between_trials = n_steps
        for coord, replica in self.replicas.items():
            replica.nsteps = self.nSteps_between_trials

    @staticmethod
    def spline(dg):
        """
        calculates the value of the spline function depending on the deviation dg from the grid point

        # Todo: PUT SOMEWHERE ELSE OR NUMPY? : numpy.interp¶.

        Parameters
        ----------
        dg:float
             deviation from gridpoint (absolute value)

        Returns
        -------
        float
            value of spline (float)
        """
        if dg < 0.0:
            print("distance smaller than 0")
        elif dg < 1.0:
            return 1.0 - 3.0 * dg * dg + 2 * dg * dg * dg
        else:
            return 0.0
