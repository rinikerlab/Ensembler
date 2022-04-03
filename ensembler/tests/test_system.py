import os
import tempfile
import unittest

import numpy as np

from ensembler import samplers
from ensembler import potentials
from ensembler import system
from ensembler.util import dataStructure as data


class test_System(unittest.TestCase):
    system_class = system.system
    tmp_test_dir: str = None

    def setUp(self) -> None:
        test_dir = os.getcwd() + "/tests_out"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if __class__.tmp_test_dir is None:
            __class__.tmp_test_dir = tempfile.mkdtemp(dir=test_dir, prefix="tmp_test_system")
        _, self.tmp_out_path = tempfile.mkstemp(prefix="test_" + self.system_class.name, suffix=".obj", dir=__class__.tmp_test_dir)

        self.sampler = samplers.stochastic.metropolisMonteCarloIntegrator()
        self.pot = potentials.OneD.harmonicOscillatorPotential()

    def test_system_constructor(self):
        self.system_class(potential=self.pot, sampler=self.sampler)

    def test_system_constructor_detail(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """

        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(
            position=[0.1],
            temperature=temperature,
            total_system_energy=0.005000000000000001,
            total_potential_energy=0.005000000000000001,
            total_kinetic_energy=np.nan,
            dhdpos=[[np.nan]],
            velocity=np.nan,
        )  # Monte carlo does not use dhdpos or velocity

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        curState = sys.current_state

        # check attributes
        self.assertEqual(
            self.pot.constants[self.pot.nDimensions], sys.nDimensions, msg="Dimensionality was not the same for system and potential!"
        )
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")
        # print(curState)
        # check current state intialisation
        self.assertEqual(expected_state.position, curState.position, msg="The initialised Position is not correct!")
        self.assertEqual(expected_state.temperature, curState.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            expected_state.total_system_energy, curState.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            expected_state.total_potential_energy,
            curState.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertEqual(
            np.isnan(expected_state.total_kinetic_energy),
            np.isnan(curState.total_kinetic_energy),
            msg="The initialised total_kinetic_energy is not correct!",
        )
        self.assertEqual(np.isnan(expected_state.velocity), np.isnan(curState.velocity), msg="The initialised velocity is not correct!")

    def test_append_state(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """

        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        expected_state = data.basicState(
            position=newPosition,
            temperature=temperature,
            total_system_energy=62.5,
            total_potential_energy=50.0,
            total_kinetic_energy=12.5,
            dhdpos=3,
            velocity=newVelocity,
        )
        # potential: _perturbedPotentialCls, samplers: _samplerCls, conditions: Iterable[Condition] = [],
        # temperature: float = 298.0, position:(Iterable[Number] or float
        sys = self.system_class(potential=self.pot, sampler=self.sampler, conditions=[], temperature=temperature, start_position=position)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces)
        curState = sys.current_state

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            curState.total_system_energy, expected_state.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertAlmostEqual(
            curState.total_kinetic_energy, expected_state.total_kinetic_energy, msg="The initialised total_kinetic_energy is not correct!"
        )
        # self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The initialised velocity is not correct!")

    def test_revertStep(self):
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        newPosition2 = 13
        newVelocity2 = -4
        newForces2 = 8

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces)
        expected_state = sys.current_state
        sys.append_state(new_position=newPosition2, new_velocity=newVelocity2, new_forces=newForces2)
        not_expected_state = sys.current_state
        sys.revert_step()
        curState = sys.current_state

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The current Position is not equal to the one two steps before!")
        self.assertEqual(
            curState.temperature, expected_state.temperature, msg="The current temperature is not equal to the one two steps before!"
        )
        self.assertAlmostEqual(
            curState.total_system_energy,
            expected_state.total_system_energy,
            msg="The current total_system_energy is not equal to the one two steps before!",
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The current total_potential_energy is not equal to the one two steps before!",
        )
        self.assertEqual(
            np.isnan(curState.total_kinetic_energy),
            np.isnan(expected_state.total_kinetic_energy),
            msg="The current total_kinetic_energy is not equal to the one two steps before!",
        )
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The current dhdpos is not equal to the one two steps before!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The current velocity is not equal to the one two steps before!")

        # check that middle step is not sames
        self.assertNotEqual(curState.position, not_expected_state.position, msg="The not expected Position equals the current one!")
        self.assertEqual(curState.temperature, not_expected_state.temperature, msg="The not expected temperature equals the current one")
        self.assertNotAlmostEqual(
            curState.total_system_energy,
            not_expected_state.total_system_energy,
            msg="The not expected total_system_energy equals the current one",
        )
        self.assertNotAlmostEqual(
            curState.total_potential_energy,
            not_expected_state.total_potential_energy,
            msg="The not expected total_potential_energy equals the current one",
        )
        self.assertEqual(
            np.isnan(curState.total_kinetic_energy),
            np.isnan(not_expected_state.total_kinetic_energy),
            msg="The not expected total_kinetic_energy equals the current one",
        )
        self.assertNotEqual(curState.dhdpos, not_expected_state.dhdpos, msg="The not expected dhdpos, equals the current one")
        self.assertNotEqual(curState.velocity, not_expected_state.velocity, msg="The not expected velocity equals the current one")

    def test_propergate(self):
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=0.005000000000000001,
            total_potential_energy=0.005000000000000001,
            total_kinetic_energy=np.nan,
            dhdpos=np.nan,
            velocity=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        initialState = sys.current_state
        sys.propagate()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )

    def test_simulate(self):
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        steps = 100

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        init_state = sys.current_state
        sys.simulate(
            steps=steps, init_system=False, withdraw_traj=True
        )  # withdrawTraj is needed in the context because of the interaction between different Tests
        trajectory = sys.trajectory

        old_frame = trajectory.iloc[0]

        # print(old_frame)
        # print(init_state)

        # Check that the first frame is the initial state!
        self.assertEqual(
            init_state.position,
            old_frame.position,
            msg="The initial state does not equal the frame 0 after propergating in attribute: Position!",
        )
        self.assertEqual(
            init_state.temperature,
            old_frame.temperature,
            msg="The initial state does not equal the frame 0 after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            init_state.total_potential_energy,
            old_frame.total_potential_energy,
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_potential_energy!",
        )
        self.assertAlmostEqual(
            np.isnan(init_state.total_kinetic_energy),
            np.isnan(old_frame.total_kinetic_energy),
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_kinetic_energy!",
        )
        self.assertEqual(
            np.isnan(init_state.dhdpos),
            np.isnan(old_frame.dhdpos),
            msg="The initial state does not equal the frame 0 after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            np.isnan(init_state.velocity),
            np.isnan(old_frame.velocity),
            msg="The initial state does not equal the frame 0 after propergating in attribute: velocity!",
        )

        # check that the frames are all different from each other.
        for ind, frame in list(trajectory.iterrows())[1:]:
            # print()
            # print(ind, frame)
            # check that middle step is not sames
            self.assertNotEqual(
                old_frame.position,
                frame.position,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: Position!",
            )
            self.assertEqual(
                old_frame.temperature,
                frame.temperature,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: temperature!",
            )  # due to samplers
            self.assertNotAlmostEqual(
                old_frame.total_potential_energy,
                frame.total_potential_energy,
                msg="The frame "
                + str(ind)
                + " equals the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_potential_energy!",
            )
            self.assertEqual(
                np.isnan(old_frame.total_kinetic_energy),
                np.isnan(frame.total_kinetic_energy),
                msg="The frame "
                + str(ind)
                + " equals not the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_kinetic_energy!",
            )  # due to samplers
            self.assertNotEqual(
                old_frame.dhdpos,
                frame.dhdpos,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: dhdpos!",
            )
            self.assertEqual(
                np.isnan(old_frame.velocity),
                np.isnan(frame.velocity),
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: velocity!",
            )  # due to samplers
            old_frame = frame

    def test_applyConditions(self):
        """
        NOT IMPLEMENTED!
        """
        pass

    def test_initVel(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """

        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        expected_state = data.basicState(
            position=newPosition,
            temperature=temperature,
            total_system_energy=62.5,
            total_potential_energy=50.0,
            total_kinetic_energy=12.5,
            dhdpos=3,
            velocity=newVelocity,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        sys._init_velocities()

        cur_velocity = sys._currentVelocities
        # print(cur_velocity)

        self.assertIsInstance(cur_velocity, float, msg="Velocity has not the correcttype!")

    def test_updateTemp(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def test_updateEne(self):
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=0.005000000000000001,
            total_potential_energy=0.005000000000000001,
            total_kinetic_energy=np.nan,
            dhdpos=np.nan,
            velocity=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        initialState = sys.current_state
        sys.propagate()
        sys._update_energies()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertNotAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertAlmostEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )

    def test_totPot(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        temperature = 300
        position = [1]
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=0.005000000000000001,
            total_potential_energy=0.005000000000000001,
            total_kinetic_energy=0,
            dhdpos=None,
            velocity=None,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        self.assertAlmostEqual(sys.calculate_total_potential_energy(), 0.5, msg="The initialised total_potential_energy is not correct!")

    def test_totKin(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        conditions = []
        temperature = 300
        position = [1]
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=0.005000000000000001,
            total_potential_energy=0.005000000000000001,
            total_kinetic_energy=np.nan,
            dhdpos=np.nan,
            velocity=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        self.assertEqual(
            np.isnan(sys.calculate_total_kinetic_energy()), np.isnan(np.nan), msg="The initialised total_kinetic_energy is not correct!"
        )

        newPosition = 10
        newVelocity = -5
        newForces = 3
        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces)
        self.assertAlmostEqual(sys.calculate_total_potential_energy(), 50.0, msg="The initialised total_potential_energy is not correct!")

    def test_setTemperature(self):
        conditions = []
        temperature = 300
        temperature2 = 600
        position = [0.1]
        mass = [1]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        sys._currentVelocities = 100
        sys.update_current_state()
        initialState = sys.current_state
        sys.set_temperature(temperature2)

        # check that middle step is not sames
        self.assertEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertNotEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does equal the currentState after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        # self.assertNotAlmostEqual(sys._currentTotKin, initialState.total_kinetic_energy,
        #                       msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!")
        self.assertEqual(
            np.isnan(sys._currentForce),
            np.isnan(initialState.dhdpos),
            msg="The initialState equals the currentState after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            sys._currentVelocities,
            initialState.velocity,
            msg="The initialState does equal the currentState after propergating in attribute: velocity!",
        )

    def test_get_Pot(self):
        conditions = []
        temperature = 300
        position = 0.1
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=0.005,
            total_potential_energy=0.005,
            total_kinetic_energy=0,
            dhdpos=None,
            velocity=None,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        self.assertEqual(0.005000000000000001, sys.total_potential_energy, msg="Could not get the correct Pot Energy!")

    def test_get_Trajectory(self):
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        sys.simulate(steps=10)
        traj_pd = sys.trajectory

    def test_save_obj_str(self):
        path = self.tmp_out_path
        out_path = self.system_class(potential=self.pot, sampler=self.sampler).save(path=path)
        print(out_path)

    def test_load_str_path(self):
        path = self.tmp_out_path
        out_path = self.system_class(potential=self.pot, sampler=self.sampler).save(path=path)

        cls = self.system_class.load(path=out_path)
        print(cls)


class test_perturbedSystem1D(test_System):
    system_class = system.perturbed_system.perturbedSystem
    tmp_test_dir: str = None

    def setUp(self) -> None:
        test_dir = os.getcwd() + "/tests_out"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if __class__.tmp_test_dir is None:
            __class__.tmp_test_dir = tempfile.mkdtemp(dir=test_dir, prefix="tmp_test_perturbedSystem")
        _, self.tmp_out_path = tempfile.mkstemp(prefix="test_" + self.system_class.name, suffix=".obj", dir=__class__.tmp_test_dir)

        self.sampler = samplers.stochastic.metropolisMonteCarloIntegrator()
        ha = potentials.OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = potentials.OneD.harmonicOscillatorPotential(x_shift=5)
        self.pot = potentials.OneD.linearCoupledPotentials(Va=ha, Vb=hb, lam=1.0)

    def test_system_constructor(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        lam = 0
        conditions = []
        temperature = 300
        position = 0
        mass = [1]
        expected_state = data.lambdaState(
            position=0,
            temperature=temperature,
            lam=0.0,
            total_system_energy=12.5,
            total_potential_energy=12.5,
            total_kinetic_energy=np.nan,
            dhdpos=np.nan,
            velocity=np.nan,
            dhdlam=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        curState = sys.current_state

        # check attributes
        self.assertEqual(
            self.pot.constants[self.pot.nDimensions], sys.nDimensions, msg="Dimensionality was not the same for system and potential!"
        )
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")
        # print(curState)
        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            curState.total_system_energy, expected_state.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertEqual(
            np.isnan(curState.total_kinetic_energy),
            np.isnan(expected_state.total_kinetic_energy),
            msg="The initialised total_kinetic_energy is not correct!",
        )
        # self.assertEqual(np.isnan(curState.dhdpos), np.isnan(expected_state.dhdpos), msg="The initialised dhdpos is not correct!")
        self.assertEqual(np.isnan(curState.velocity), np.isnan(expected_state.velocity), msg="The initialised velocity is not correct!")
        self.assertEqual(np.isnan(curState.lam), np.isnan(expected_state.lam), msg="The initialised lam is not correct!")
        # self.assertEqual(np.isnan(curState.dhdlam), np.isnan(expected_state.dhdlam), msg="The initialised dHdlam is not correct!")

    def test_system_constructor_detail(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(
            position=[0.1],
            temperature=temperature,
            total_system_energy=12.005,
            total_potential_energy=12.005,
            total_kinetic_energy=np.nan,
            dhdpos=[[np.nan]],
            velocity=np.nan,
        )  # Monte carlo does not use dhdpos or velocity

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        curState = sys.current_state

        # check attributes
        self.assertEqual(
            self.pot.constants[self.pot.nDimensions], sys.nDimensions, msg="Dimensionality was not the same for system and potential!"
        )
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")
        # print(curState)
        # check current state intialisation
        self.assertEqual(expected_state.position, curState.position, msg="The initialised Position is not correct!")
        self.assertEqual(expected_state.temperature, curState.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            expected_state.total_system_energy, curState.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            expected_state.total_potential_energy,
            curState.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertEqual(
            np.isnan(expected_state.total_kinetic_energy),
            np.isnan(curState.total_kinetic_energy),
            msg="The initialised total_kinetic_energy is not correct!",
        )
        self.assertEqual(np.isnan(expected_state.velocity), np.isnan(curState.velocity), msg="The initialised velocity is not correct!")

    def test_append_state(self):

        lam = 0
        temperature = 300
        position = 0

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1.0

        expected_state = data.lambdaState(
            position=newPosition,
            temperature=temperature,
            lam=newLam,
            total_system_energy=125.0,
            total_potential_energy=112.5,
            total_kinetic_energy=12.5,
            dhdpos=newForces,
            velocity=newVelocity,
            dhdlam=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_lambda=newLam)
        curState = sys.current_state

        # check current state intialisation
        self.assertEqual(sys._currentPosition, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            curState.total_system_energy, expected_state.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertAlmostEqual(
            curState.total_kinetic_energy, expected_state.total_kinetic_energy, msg="The initialised total_kinetic_energy is not correct!"
        )
        # self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(np.isnan(curState.velocity), np.isnan(expected_state.velocity), msg="The initialised velocity is not correct!")
        self.assertEqual(curState.lam, expected_state.lam, msg="The initialised lam is not correct!")
        # self.assertEqual(np.isnan(curState.dhdlam), np.isnan(expected_state.dhdlam), msg="The initialised dHdlam is not correct!")

    def test_revertStep(self):
        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1.0

        newPosition2 = 13
        newVelocity2 = -4
        newForces2 = 8
        newLam2 = 0.5

        lam = 0
        conditions = []
        temperature = 300
        position = [0]
        mass = [1]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_lambda=newLam)
        expected_state = sys.current_state
        sys.append_state(new_position=newPosition2, new_velocity=newVelocity2, new_forces=newForces2, new_lambda=newLam2)
        not_expected_state = sys.current_state
        sys.revert_step()
        curState = sys.current_state

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The current Position is not equal to the one two steps before!")
        self.assertEqual(
            curState.temperature, expected_state.temperature, msg="The current temperature is not equal to the one two steps before!"
        )
        self.assertAlmostEqual(
            curState.total_system_energy,
            expected_state.total_system_energy,
            msg="The current total_system_energy is not equal to the one two steps before!",
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The current total_potential_energy is not equal to the one two steps before!",
        )
        self.assertAlmostEqual(
            curState.total_kinetic_energy,
            expected_state.total_kinetic_energy,
            msg="The current total_kinetic_energy is not equal to the one two steps before!",
        )
        # self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The current dhdpos is not equal to the one two steps before!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The current velocity is not equal to the one two steps before!")
        self.assertEqual(curState.lam, expected_state.lam, msg="The current lam is not equal to the one two steps before!")
        self.assertEqual(np.isnan(curState.dhdlam), np.isnan(expected_state.dhdlam), msg="The initialised dHdlam is not correct!")

        # check that middle step is not sames
        self.assertNotEqual(curState.position, not_expected_state.position, msg="The not expected Position equals the current one!")
        self.assertEqual(curState.temperature, not_expected_state.temperature, msg="The not expected temperature equals the current one")
        self.assertNotAlmostEqual(
            curState.total_system_energy,
            not_expected_state.total_system_energy,
            msg="The not expected total_system_energy equals the current one",
        )
        self.assertNotAlmostEqual(
            curState.total_potential_energy,
            not_expected_state.total_potential_energy,
            msg="The not expected total_potential_energy equals the current one",
        )
        self.assertNotAlmostEqual(
            curState.total_kinetic_energy,
            not_expected_state.total_kinetic_energy,
            msg="The not expected total_kinetic_energy equals the current one",
        )
        # self.assertNotEqual(curState.dhdpos, not_expected_state.dhdpos, msg="The not expected dhdpos, equals the current one")
        self.assertNotEqual(curState.velocity, not_expected_state.velocity, msg="The not expected velocity equals the current one")
        self.assertNotEqual(curState.lam, not_expected_state.lam, msg="The not expected lam equals the current one")
        self.assertEqual(np.isnan(curState.dhdlam), np.isnan(expected_state.dhdlam), msg="The initialised dHdlam is not correct!")

    def test_propergate(self):
        lam = 0
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        initialState = sys.current_state
        sys.propagate()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propagating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )
        self.assertEqual(
            sys._currentLambda,
            initialState.lam,
            msg="The initialState does not equal the currentState after propergating in attribute: lam!",
        )
        self.assertEqual(
            np.isnan(sys._currentdHdLambda),
            np.isnan(initialState.dhdlam),
            msg="The initialState does not equal the currentState after propergating in attribute: dHdLam!",
        )

    def test_simulate(self):
        lam = 0
        steps = 100

        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        init_state = sys.current_state
        sys.simulate(
            steps=steps, init_system=False, withdraw_traj=True
        )  # withdrawTraj is needed in the context because of the interaction between different Tests
        trajectory = sys.trajectory

        old_frame = trajectory.iloc[0]
        # Check that the first frame is the initial state!
        self.assertListEqual(
            list(init_state.position),
            list(old_frame.position),
            msg="The initial state does not equal the frame 0 after propergating in attribute: Position!",
        )
        self.assertEqual(
            init_state.temperature,
            old_frame.temperature,
            msg="The initial state does not equal the frame 0 after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            init_state.total_potential_energy,
            old_frame.total_potential_energy,
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_potential_energy!",
        )
        self.assertAlmostEqual(
            np.isnan(init_state.total_kinetic_energy),
            np.isnan(old_frame.total_kinetic_energy),
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_kinetic_energy!",
        )
        self.assertEqual(
            np.isnan(init_state.dhdpos),
            np.isnan(old_frame.dhdpos),
            msg="The initial state does not equal the frame 0 after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            np.isnan(init_state.velocity),
            np.isnan(old_frame.velocity),
            msg="The initial state does not equal the frame 0 after propergating in attribute: velocity!",
        )
        self.assertEqual(
            init_state.lam, old_frame.lam, msg="The initial state does not equal the frame 0 after propergating in attribute: lam!"
        )
        self.assertEqual(
            np.isnan(init_state.dhdlam),
            np.isnan(old_frame.dhdlam),
            msg="The initial state does not equal the frame 0 after propergating in attribute: dhdLam!",
        )

        # check that the frames are all different from each other.
        for ind, frame in list(trajectory.iterrows())[1:]:
            # check that middle step is not sames
            self.assertNotEqual(
                old_frame.position,
                frame.position,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: Position!",
            )
            self.assertEqual(
                old_frame.temperature,
                frame.temperature,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: temperature!",
            )  # due to samplers
            self.assertNotAlmostEqual(
                old_frame.total_potential_energy,
                frame.total_potential_energy,
                msg="The frame "
                + str(ind)
                + " equals the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_potential_energy!",
            )
            self.assertAlmostEqual(
                np.isnan(old_frame.total_kinetic_energy),
                np.isnan(frame.total_kinetic_energy),
                msg="The frame "
                + str(ind)
                + " equals the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_kinetic_energy!",
            )  # due to samplers
            self.assertNotEqual(
                old_frame.dhdpos,
                frame.dhdpos,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: dhdpos!",
            )
            self.assertEqual(
                np.isnan(old_frame.velocity),
                np.isnan(frame.velocity),
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: velocity!",
            )  # due to samplers
            self.assertEqual(
                init_state.lam,
                old_frame.lam,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: lam!",
            )
            self.assertEqual(
                np.isnan(init_state.dhdlam),
                np.isnan(old_frame.dhdlam),
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: dhdLam!",
            )
            old_frame = frame

    def test_applyConditions(self):
        """
        NOT IMPLEMENTED!
        """
        pass

    def test_initVel(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        lam = 0

        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        sys._init_velocities()

        cur_velocity = sys._currentVelocities
        # print(cur_velocity)
        expected_vel = np.float64(-2.8014573319669176)
        self.assertEqual(type(cur_velocity), type(expected_vel), msg="Velocity has not the correcttype!")

    def test_updateTemp(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def test_updateEne(self):
        lam = 0
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        initialState = sys.current_state
        sys.propagate()
        sys._update_energies()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertNotAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )

    def test_totPot(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """

        lam = 0
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        self.assertAlmostEqual(sys.calculate_total_potential_energy(), 12.5, msg="The initialised total_potential_energy is not correct!")

    def test_totKin(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        lam = 0

        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        self.assertTrue(np.isnan(sys.calculate_total_kinetic_energy()), msg="The initialised total_potential_energy is not correct!")

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1
        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_lambda=newLam)
        self.assertAlmostEqual(sys.calculate_total_kinetic_energy(), 12.5, msg="The initialised total_potential_energy is not correct!")

    def test_setTemperature(self):
        lam = 0
        temperature = 300
        temperature2 = 600
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        sys._currentVelocities = 100
        sys.update_current_state()
        initialState = sys.current_state
        sys.set_temperature(temperature2)

        # check that middle step is not sames
        self.assertListEqual(
            list(sys._currentPosition),
            list(initialState.position),
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertNotEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does equal the currentState after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertNotAlmostEqual(
            sys._currentTotKin,
            initialState.total_kinetic_energy,
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentForce),
            np.isnan(initialState.dhdpos),
            msg="The initialState equals the currentState after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            sys._currentVelocities,
            initialState.velocity,
            msg="The initialState does equal the currentState after propergating in attribute: velocity!",
        )

    def test_get_Pot(self):
        lam = 0
        temperature = 300
        position = [5]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, lam=lam)
        self.assertEqual(0.0, sys.total_potential_energy, msg="Could not get the correct Pot Energy!")


class test_edsSystem1D(test_System):
    system_class = system.eds_system.edsSystem
    tmp_test_dir: str = None

    def setUp(self) -> None:
        test_dir = os.getcwd() + "/tests_out"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if __class__.tmp_test_dir is None:
            __class__.tmp_test_dir = tempfile.mkdtemp(dir=test_dir, prefix="tmp_test_eds_system")
        _, self.tmp_out_path = tempfile.mkstemp(prefix="test_" + self.system_class.name, suffix=".obj", dir=__class__.tmp_test_dir)

        self.sampler = samplers.stochastic.metropolisMonteCarloIntegrator()
        self.pot = potentials.OneD.envelopedPotential()

    def test_system_constructor(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        s = 1
        conditions = []
        temperature = 300
        position = 0
        mass = [1]
        expected_state = data.envelopedPStstate(
            position=0,
            temperature=temperature,
            s=1.0,
            eoff=[0, 0],
            total_system_energy=-0.011047744848593777,
            total_potential_energy=-0.011047744848593777,
            total_kinetic_energy=np.nan,
            dhdpos=np.nan,
            velocity=np.nan,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        curState = sys.current_state

        # check attributes
        self.assertEqual(
            self.pot.constants[self.pot.nDimensions], sys.nDimensions, msg="Dimensionality was not the same for system and potential!"
        )
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")
        # print(curState)
        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            curState.total_system_energy, expected_state.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertEqual(
            np.isnan(curState.total_kinetic_energy),
            np.isnan(expected_state.total_kinetic_energy),
            msg="The initialised total_kinetic_energy is not correct!",
        )
        # self.assertEqual(np.isnan(curState.dhdpos), np.isnan(expected_state.dhdpos), msg="The initialised dhdpos is not correct!")
        self.assertEqual(np.isnan(curState.velocity), np.isnan(expected_state.velocity), msg="The initialised velocity is not correct!")
        self.assertEqual(curState.s, expected_state.s, msg="The initialised s is not correct!")
        self.assertEqual(curState.eoff, expected_state.eoff, msg="The initialised Eoff is not correct!")

    def test_system_constructor_detail(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """

        conditions = []
        temperature = 300
        position = 0.1
        mass = [1]
        expected_state = data.basicState(
            position=position,
            temperature=temperature,
            total_system_energy=-0.009884254671918117,
            total_potential_energy=-0.009884254671918117,
            total_kinetic_energy=np.nan,
            dhdpos=np.array(-0.0556779),
            velocity=np.nan,
        )  # Monte carlo does not use dhdpos or velocity

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature)
        curState = sys.current_state
        print(curState)
        # check attributes
        self.assertEqual(
            self.pot.constants[self.pot.nDimensions], sys.nDimensions, msg="Dimensionality was not the same for system and potential!"
        )
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")
        # print(curState)
        # check current state intialisation
        self.assertEqual(expected_state.position, curState.position, msg="The initialised Position is not correct!")
        self.assertEqual(expected_state.temperature, curState.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            expected_state.total_system_energy, curState.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            expected_state.total_potential_energy,
            curState.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertEqual(
            np.isnan(expected_state.total_kinetic_energy),
            np.isnan(curState.total_kinetic_energy),
            msg="The initialised total_kinetic_energy is not correct!",
        )
        self.assertEqual(np.isnan(expected_state.velocity), np.isnan(curState.velocity), msg="The initialised velocity is not correct!")

    def test_append_state(self):
        temperature = 300
        position = 0
        s = 1.0
        Eoff = [0, 0]

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newEoff = [1, 1]
        newS = [2]

        expected_state = data.envelopedPStstate(
            position=newPosition,
            temperature=temperature,
            s=newS,
            eoff=newEoff,
            total_system_energy=36.99999999999157,
            total_potential_energy=24.499999999991577,
            total_kinetic_energy=12.5,
            dhdpos=newForces,
            velocity=newVelocity,
        )

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_s=newS, new_eoff=newEoff)
        curState = sys.current_state

        # check current state intialisation
        self.assertEqual(sys._currentPosition, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(
            curState.total_system_energy, expected_state.total_system_energy, msg="The initialised total_system_energy is not correct!"
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The initialised total_potential_energy is not correct!",
        )
        self.assertAlmostEqual(
            curState.total_kinetic_energy, expected_state.total_kinetic_energy, msg="The initialised total_kinetic_energy is not correct!"
        )
        # self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(np.isnan(curState.velocity), np.isnan(expected_state.velocity), msg="The initialised velocity is not correct!")
        self.assertEqual(curState.s, expected_state.s, msg="The initialised s is not correct!")
        self.assertEqual(curState.eoff, expected_state.eoff, msg="The initialised Eoff is not correct!")

    def test_revertStep(self):
        newPosition = 10
        newVelocity = -5
        newForces = 3
        newS = 1.0
        newEoff = [1, 1]

        newPosition2 = 13
        newVelocity2 = -4
        newForces2 = 8
        newS2 = 0.5
        newEoff2 = [2, 2]

        integ = samplers.stochastic.metropolisMonteCarloIntegrator()
        ha = potentials.OneD.harmonicOscillatorPotential(x_shift=-5)
        hb = potentials.OneD.harmonicOscillatorPotential(x_shift=5)
        s = 1
        pot = potentials.OneD.exponentialCoupledPotentials(Va=ha, Vb=hb, s=1)
        conditions = []
        temperature = 300
        position = [0]
        mass = [1]

        sys = self.system_class(potential=pot, sampler=integ, start_position=position, temperature=temperature, eds_s=s)

        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_s=newS, new_eoff=newEoff)
        expected_state = sys.current_state
        sys.append_state(new_position=newPosition2, new_velocity=newVelocity2, new_forces=newForces2, new_s=newS2, new_eoff=newEoff2)
        not_expected_state = sys.current_state
        print(len(sys._trajectory), sys._trajectory)
        sys.revert_step()
        curState = sys.current_state
        print(curState)
        print(not_expected_state)

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The current Position is not equal to the one two steps before!")
        self.assertEqual(
            curState.temperature, expected_state.temperature, msg="The current temperature is not equal to the one two steps before!"
        )
        self.assertAlmostEqual(
            curState.total_system_energy,
            expected_state.total_system_energy,
            msg="The current total_system_energy is not equal to the one two steps before!",
        )
        self.assertAlmostEqual(
            curState.total_potential_energy,
            expected_state.total_potential_energy,
            msg="The current total_potential_energy is not equal to the one two steps before!",
        )
        self.assertAlmostEqual(
            curState.total_kinetic_energy,
            expected_state.total_kinetic_energy,
            msg="The current total_kinetic_energy is not equal to the one two steps before!",
        )
        # self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The current dhdpos is not equal to the one two steps before!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The current velocity is not equal to the one two steps before!")
        self.assertEqual(curState.s, expected_state.s, msg="The current s is not equal to the one two steps before!")
        np.testing.assert_almost_equal(
            curState.eoff, expected_state.eoff, err_msg="The initialised Eoff is not correct as not equal to two steps before!"
        )

        # check that middle step is not sames
        self.assertNotEqual(curState.position, not_expected_state.position, msg="The not expected Position equals the current one!")
        self.assertEqual(curState.temperature, not_expected_state.temperature, msg="The not expected temperature equals the current one")
        self.assertNotAlmostEqual(
            curState.total_system_energy,
            not_expected_state.total_system_energy,
            msg="The not expected total_system_energy equals the current one",
        )
        self.assertNotAlmostEqual(
            curState.total_potential_energy,
            not_expected_state.total_potential_energy,
            msg="The not expected total_potential_energy equals the current one",
        )
        self.assertNotAlmostEqual(
            curState.total_kinetic_energy,
            not_expected_state.total_kinetic_energy,
            msg="The not expected total_kinetic_energy equals the current one",
        )
        # self.assertNotEqual(curState.dhdpos, not_expected_state.dhdpos, msg="The not expected dhdpos, equals the current one")
        self.assertNotEqual(curState.velocity, not_expected_state.velocity, msg="The not expected velocity equals the current one")
        self.assertNotEqual(curState.s, not_expected_state.s, msg="The not expected lam equals the current one")
        self.assertNotEqual(curState.eoff, not_expected_state.eoff, msg="The initialised Eoff is not correct!")

    def test_propergate(self):
        temperature = 300
        position = [0]
        s = 1

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        initialState = sys.current_state
        sys.propagate()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propagating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )
        self.assertEqual(
            sys._currentEdsS, initialState.s, msg="The initialState does not equal the currentState after propergating in attribute: s!"
        )
        np.testing.assert_almost_equal(
            sys._currentEdsEoffs,
            initialState.eoff,
            err_msg="The initialState does not equal the currentState after propergating in attribute: Eoff!",
        )

    def test_simulate(self):
        s = 1
        steps = 100

        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        init_state = sys.current_state
        sys.simulate(
            steps=steps, init_system=False, withdraw_traj=True
        )  # withdrawTraj is needed in the context because of the interaction between different Tests
        trajectory = sys.trajectory

        old_frame = trajectory.iloc[0]
        # Check that the first frame is the initial state!
        self.assertListEqual(
            list(init_state.position),
            list(old_frame.position),
            msg="The initial state does not equal the frame 0 after propergating in attribute: Position!",
        )
        self.assertEqual(
            init_state.temperature,
            old_frame.temperature,
            msg="The initial state does not equal the frame 0 after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            init_state.total_potential_energy,
            old_frame.total_potential_energy,
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_potential_energy!",
        )
        self.assertAlmostEqual(
            np.isnan(init_state.total_kinetic_energy),
            np.isnan(old_frame.total_kinetic_energy),
            msg="The initial state does not equal the frame 0 after propergating in attribute: total_kinetic_energy!",
        )
        self.assertEqual(
            np.isnan(init_state.dhdpos),
            np.isnan(old_frame.dhdpos),
            msg="The initial state does not equal the frame 0 after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            np.isnan(init_state.velocity),
            np.isnan(old_frame.velocity),
            msg="The initial state does not equal the frame 0 after propergating in attribute: velocity!",
        )
        self.assertEqual(init_state.s, old_frame.s, msg="The initial state does not equal the frame 0 after propergating in attribute: s!")
        np.testing.assert_almost_equal(
            init_state.eoff, old_frame.eoff, err_msg="The initial state does not equal the frame 0 after propergating in attribute: Eoff!"
        )

        # check that the frames are all different from each other.
        for ind, frame in list(trajectory.iterrows())[1:]:
            # check that middle step is not sames
            self.assertNotEqual(
                old_frame.position,
                frame.position,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: Position!",
            )
            self.assertEqual(
                old_frame.temperature,
                frame.temperature,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: temperature!",
            )  # due to samplers
            self.assertNotAlmostEqual(
                old_frame.total_potential_energy,
                frame.total_potential_energy,
                msg="The frame "
                + str(ind)
                + " equals the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_potential_energy!",
            )
            self.assertAlmostEqual(
                np.isnan(old_frame.total_kinetic_energy),
                np.isnan(frame.total_kinetic_energy),
                msg="The frame "
                + str(ind)
                + " equals the frame  "
                + str(ind + 1)
                + " after propergating in attribute: total_kinetic_energy!",
            )  # due to samplers
            self.assertNotEqual(
                old_frame.dhdpos,
                frame.dhdpos,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: dhdpos!",
            )
            self.assertEqual(
                np.isnan(old_frame.velocity),
                np.isnan(frame.velocity),
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: velocity!",
            )  # due to samplers
            self.assertEqual(
                init_state.s,
                old_frame.s,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: s!",
            )
            self.assertEqual(
                init_state.eoff,
                old_frame.eoff,
                msg="The frame " + str(ind) + " equals the frame  " + str(ind + 1) + " after propergating in attribute: Eoff!",
            )
            old_frame = frame

    def test_applyConditions(self):
        """
        NOT IMPLEMENTED!
        """
        pass

    def test_initVel(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        s = 1
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        sys._init_velocities()

        cur_velocity = sys._currentVelocities
        expected_vel = [0.19334311622217965, 1.2272590394440765]
        self.assertEqual(type(cur_velocity), type(expected_vel), msg="Velocity has not the correcttype!")

    def test_updateTemp(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def test_updateEne(self):
        s = 1
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        initialState = sys.current_state
        sys.propagate()
        sys._update_energies()

        # check that middle step is not sames
        self.assertNotEqual(
            sys._currentPosition,
            initialState.position,
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does not equal the currentState after propergating in attribute: temperature!",
        )
        self.assertNotAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentTotKin),
            np.isnan(initialState.total_kinetic_energy),
            msg="The initialState  does equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertNotEqual(
            sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!"
        )
        self.assertEqual(
            np.isnan(sys._currentVelocities),
            np.isnan(initialState.velocity),
            msg="The initialState does not equal the currentState after propergating in attribute: velocity!",
        )

    def test_totPot(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        s = 1
        temperature = 300
        position = [12]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        self.assertAlmostEqual(
            sys.calculate_total_potential_energy(), 40.49999999999998, msg="The initialised total_potential_energy is not correct!"
        )

    def test_totKin(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        s = 1
        temperature = 300
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        self.assertTrue(np.isnan(sys.calculate_total_kinetic_energy()), msg="The initialised total_potential_energy is not correct!")

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newS = 2
        newEoff = [0, 0]
        sys.append_state(new_position=newPosition, new_velocity=newVelocity, new_forces=newForces, new_s=newS, new_eoff=newEoff)
        self.assertAlmostEqual(sys.calculate_total_kinetic_energy(), 12.5, msg="The initialised total_potential_energy is not correct!")

    def test_setTemperature(self):
        s = 1
        temperature = 300
        temperature2 = 600
        position = [0]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        sys._currentVelocities = 100
        sys.update_current_state()
        initialState = sys.current_state
        sys.set_temperature(temperature2)

        # check that middle step is not sames
        self.assertListEqual(
            list(sys.position),
            list(initialState.position),
            msg="The initialState equals the currentState after propergating in attribute: Position!",
        )
        self.assertNotEqual(
            sys._currentTemperature,
            initialState.temperature,
            msg="The initialState does equal the currentState after propergating in attribute: temperature!",
        )
        self.assertAlmostEqual(
            sys._currentTotPot,
            initialState.total_potential_energy,
            msg="The initialState  does equal  the currentState after propergating in attribute: total_potential_energy!",
        )
        self.assertNotAlmostEqual(
            sys._currentTotKin,
            initialState.total_kinetic_energy,
            msg="The initialState  does not equal  the currentState after propergating in attribute: total_kinetic_energy!",
        )
        self.assertEqual(
            np.isnan(sys._currentForce),
            np.isnan(initialState.dhdpos),
            msg="The initialState equals the currentState after propergating in attribute: dhdpos!",
        )
        self.assertEqual(
            sys._currentVelocities,
            initialState.velocity,
            msg="The initialState does equal the currentState after propergating in attribute: velocity!",
        )

    def test_get_Pot(self):
        s = 1
        temperature = 300
        position = [5]

        sys = self.system_class(potential=self.pot, sampler=self.sampler, start_position=position, temperature=temperature, eds_s=s)
        self.assertEqual(1.9999724639297711, sys.total_potential_energy, msg="Could not get the correct Pot Energy!")


if __name__ == "__main__":
    unittest.main()
