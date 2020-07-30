import unittest
from ensembler.system import basic_system as system
from ensembler.integrator import stochastic, newtonian
from ensembler.potentials import OneD
from ensembler.ensemble import replica_exchange, _replica_graph


class test_ReplicaExchangeCls(unittest.TestCase):
    RE = _replica_graph.ReplicaExchange
    integrator = stochastic.monteCarloIntegrator()
    potential = OneD.harmonicOscillatorPotential()
    sys = system.system(potential=potential, integrator=integrator)
    group:replica_exchange.ReplicaExchange = None

    def test_setUp(self) -> None:
        self.group = None

    def test_tearDown(self) -> None:
        self.RE.replicas = {}

    def test_init_1DREnsemble(self):
        from ensembler import ensemble
        exchange_dimensions = {"temperature": range(288, 310)}
        _replica_graph.ReplicaExchange(system=self.sys, exchange_dimensions=exchange_dimensions)

    def test_init_2DREnsemble(self):
        from ensembler import ensemble
        exchange_dimensions = {"temperature": range(288, 310),
                               "mass": range(1,10)}

        _replica_graph.ReplicaExchange(system=self.sys, exchange_dimensions=exchange_dimensions)

    def test_run_1DREnsemble(self):
        from ensembler import ensemble
        exchange_dimensions = {"temperature": range(288, 310)}

        group = _replica_graph.ReplicaExchange(system=self.sys, exchange_dimensions=exchange_dimensions)
        group.run()

    def test_getTraj_1DREnsemble(self):
        replicas =22
        nsteps = 100
        group = None
        from ensembler import ensemble
        exchange_dimensions = {"temperature": range(288, 310)}

        self.group = _replica_graph.ReplicaExchange(system=self.sys, exchange_dimensions=exchange_dimensions)
        self.group.nSteps_between_trials = nsteps
        self.group.run()
        trajectories = self.group.get_trajectories()


        ##print(len(trajectories))
        ##print([len(trajectories[t]) for t in trajectories])

        self.assertEqual(len(trajectories), 22, msg="not enough trajectories were retrieved!")
        self.assertEquals([len(trajectories[t]) for t in trajectories], second=[nsteps+1 for x in range(replicas)], msg="traj lengths are not correct!")

    def test_getTotPot_1DREnsemble(self):
        replicas =22
        nsteps = 100
        from ensembler import ensemble
        exchange_dimensions = {"temperature": range(288, 310)}

        self.group = _replica_graph.ReplicaExchange(system=self.sys,exchange_dimensions=exchange_dimensions)
        self.group.nSteps_between_trials = nsteps
        self.group.run()
        totPots = self.group.get_total_energy()


        ##print(len(totPots))
        ##print(totPots)
        self.assertEqual(len(totPots), replicas, msg="not enough trajectories were retrieved!")

    def test_setPositionsList_1DREnsemble(self):
        exchange_dimensions = {"temperature": range(288, 310)}
        replicas =len(exchange_dimensions["temperature"])
        expected_pos= range(replicas)
        self.group = _replica_graph.ReplicaExchange(system=self.sys, exchange_dimensions=exchange_dimensions)

        initial_positions = sorted([self.group.replicas[replica]._currentPosition for replica in self.group.replicas])
        self.group.set_replicas_positions(expected_pos)
        setted_pos = sorted([self.group.replicas[replica]._currentPosition for replica in self.group.replicas])

        self.assertEqual(len(self.group.replicas), replicas, msg="not enough trajectories were retrieved!")
        self.assertNotEqual(initial_positions, setted_pos, msg="Setted positions are the same as before!")
        self.assertEqual(setted_pos, list(expected_pos), msg="The positions were not set correctly!")

class test_TemperatureReplicaExchangeCls(unittest.TestCase):
    TRE = replica_exchange.TemperatureReplicaExchange
    group: replica_exchange.TemperatureReplicaExchange

    def test_tearDown(self) -> None:
        setattr(self, "group", None)
        T_range=None

    def test_setUp(self) -> None:
        self.group = None
        self.TRE.replicas = {}
        T_range = None

    def test_init(self):
        integrator = stochastic.monteCarloIntegrator()
        potential = OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        replicas =22
        nsteps = 100
        T_range=range(288, 310)
        setattr(self, "group", None)
        self.group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)

    def test_run(self):
        integrator = stochastic.monteCarloIntegrator()
        potential =OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        replicas =22
        nsteps = 100
        T_range=range(288, 310)
        self.group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)
        ##print(self.group.get_Total_Energy())
        self.group.nSteps_between_trials = nsteps
        self.group.run()
        ##print(self.group.get_Total_Energy())


    def test_exchange_all(self):
        integrator = stochastic.monteCarloIntegrator()
        potential = OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        T_range=range(1, 10)
        nReplicas = len(T_range)
        positions = list([float(1) for x in range(nReplicas)])
        velocities = list([float(0) for x in range(nReplicas)])


        self.group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)
        self.group.set_replicas_positions(positions)
        self.group.set_replicas_velocities(velocities)
        self.group._defaultRandomness= lambda x,y: False

        self.group.exchange()
        all_exchanges = self.group._current_exchanges
        finpositions = list(self.group.get_replicas_positions().values())
        finvelocities = list(self.group.get_replicas_velocities().values())

        #Checking:
        ##constant params?
        self.assertEqual(len(self.group.replicas), nReplicas, msg="not enough trajectories were retrieved!")
        self.assertListEqual(finpositions, positions, msg="Positions should not change during exchange!")
        self.assertListEqual(finvelocities, velocities, msg="Velocities should not change during exchange!")
        ##exchange process
        self.assertEqual(nReplicas // 2, len(all_exchanges), msg="length of all exchanges is not correct!")
        self.assertTrue(all(list(all_exchanges.values())), msg="not all exchanges are True!!")
        del self.group
        setattr(self, "group", None)

    def test_exchange_none(self):
        """
        TODO FIX

        :return:
        """

        integrator = newtonian.positionVerletIntegrator()
        potential = OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        T_range = [1, 200, 500]
        nReplicas = len(T_range)
        print("REPS", nReplicas)
        positions = [float(x)*100 for x in range(nReplicas)]
        velocities = list([float(1) for x in range(nReplicas)])

        group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)
        print("REPS", group.nReplicas)

        #remove Randomness!
        group._defaultRandomness= lambda x,y: False
        group.set_replicas_positions(positions)
        group.set_replicas_velocities(velocities)

        #first round
        group.exchange()
        all_exchanges = group._current_exchanges
        finpositions = list(group.get_replicas_positions().values())
        finvelocities = list(group.get_replicas_velocities().values())

        #Checking:
        ##constant params?
        self.assertEqual(len(group.replicas), nReplicas, msg="not enough trajectories were retrieved!")
        self.assertListEqual(finpositions, positions, msg="Positions should not change during exchange!")
        self.assertListEqual(finvelocities, velocities, msg="Velocities should not change during exchange!")
        ##exchange process
        ##print(all_exchanges.values)
        self.assertEqual(nReplicas//2, len(all_exchanges), msg="length of all exchanges is not correct!")
        #self.assertFalse(all(list(all_exchanges.values())), msg="length of all exchanges is not correct!")
        print(group.exchange_information[["nExchange", "replicaI", "replicaJ", "TotEI", "TotEJ", "doExchange"]])

    def test_simulate_good_exchange(self):
        integrator = stochastic.monteCarloIntegrator()
        potential = OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        replicas =22
        nsteps = 100
        T_range=range(288, 310)
        self.group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)
        ##print(self.group.get_Total_Energy())
        self.group.nSteps_between_trials = nsteps
        self.group.simulate(5)
        ##print(self.group.get_Total_Energy())
        ##print("Exchanges: ", self.group.exchange_information)

    def test_simulate_bad_exchange(self):
        integrator = newtonian.positionVerletIntegrator()

        potential = OneD.harmonicOscillatorPotential()
        sys = system.system(potential=potential, integrator=integrator)

        replicas =3
        nsteps = 1

        T_range = [1, 2000, 5000]
        positions = [float(x)*100 for x in range(len(T_range))]

        self.group = replica_exchange.TemperatureReplicaExchange(system=sys, temperature_Range=T_range)

        #remove Randomness!
        self.group._defaultRandomness= lambda x,y: False
        self.group.set_replicas_positions(positions)
        self.group.nSteps_between_trials = nsteps
        self.group.set_replicas_velocities([float(-100) for x in range(replicas)])

        print("STEP:\t", 0)
        print(self.group.get_replicas_positions())
        print("ENERGY: ", self.group.get_total_energy())
        print("POSITION: ", self.group.get_replicas_positions())
        print("\n".join([str(replica.getCurrentState()) for coord, replica in self.group.replicas.items()]))


        for step in range(5):
            #self.group.run()
            self.group.simulate(1)
            self.group.set_replicas_velocities([float(10) for x in range(replicas)])
            #self.group.exchange()
            #self.group.simulate(1)
            print("STEP:\t", step)
            print("ENERGY: ", self.group.get_total_energy())
            print("\n".join([str(replica.getCurrentState()) for coord,replica in self.group.replicas.items()]))

        print("Exchanges: ", self.group.exchange_information.columns)
        print(self.group.exchange_information[["nExchange", "replicaI", "replicaJ", "TotEI", "TotEJ", "doExchange"]])

        self.assertFalse(any(list(self.group.exchange_information.doExchange)), msg="No Exchange should happen!")


if __name__ == '__main__':
    unittest.main()
