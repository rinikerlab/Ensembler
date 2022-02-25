import numpy as np
from ensembler.ensemble import replica_exchange, _replica_graph
from ensembler.samplers import stochastic
from ensembler.potentials import OneD
from ensembler.system import basic_system as system

import multiprocessing


def main():
    TRE = replica_exchange.temperatureReplicaExchange

    integrator = stochastic.metropolisMonteCarloIntegrator()
    potential = OneD.harmonicOscillatorPotential()
    sys = system.system(potential=potential, sampler=integrator)

    replicas = 2
    nsteps = 10
    T_range = np.linspace(288, 310, num=replicas)
    group = replica_exchange.temperatureReplicaExchange(system=sys, temperature_range=T_range)
    print("TotENERGY:", group.get_replica_total_energies())

    group.nSteps_between_trials = nsteps
    group._run_parallel(1)

    print("FINI: ", [traj.shape for key, traj in group.get_trajectories().items()])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
