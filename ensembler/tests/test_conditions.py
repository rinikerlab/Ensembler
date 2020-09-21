import numpy as np
import os
import tempfile
import unittest

from ensembler.conditions.box_conditions import periodicBoundaryCondition, boxBoundaryCondition
from ensembler.conditions.restrain_conditions import positionRestraintCondition

tmp_potentials = tempfile.mkdtemp(dir=os.getcwd(), prefix="tmp_test_conditions")


class boxBoundaryCondition(unittest.TestCase):
    condition_class = boxBoundaryCondition
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + condition_class.name, suffix=".obj", dir=tmp_potentials)
    boundary1D = [0, 10]
    boundary2D = [[0, 10], [0, 10]]

    def test_constructor(self):
        print(self.condition_class(boundary=self.boundary1D))

    def test_constructor2D(self):
        print(self.condition_class(boundary=self.boundary2D))

    def test_save_obj_str(self):
        path = self.tmp_out_path
        out_path = self.condition_class(self.boundary2D).save(path=path)
        print(out_path)

    def test_load_str_path(self):
        path = self.tmp_out_path
        out_path = self.condition_class(self.boundary2D).save(path=path)

        cls = self.condition_class.load(path=out_path)
        print(cls)

    def test_apply1D(self):
        cond = self.condition_class(boundary=self.boundary2D)
        cond.verbose = True

        expected_pos = [2]
        expected_vel = [0.2]
        position = [-2]
        velocity = [-0.2]

        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=velocity)

        self.assertEqual(first=corr_pos, second=expected_pos,
                         msg="The position correction for the lower bound position was wrong.")
        self.assertEqual(first=corr_vel, second=expected_vel,
                         msg="The position correction for the lower bound velocity was wrong.")

        expected_pos = [6]
        expected_vel = [-2.2]
        position = [14]
        velocity = [2.2]

        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=velocity)

        self.assertEqual(first=corr_pos, second=expected_pos,
                         msg="The position correction for the lower bound position was wrong.")
        self.assertEqual(first=corr_vel, second=expected_vel,
                         msg="The position correction for the lower bound velocity was wrong.")

    def test_apply2D(self):
        cond = self.condition_class(boundary=self.boundary2D)
        cond.verbose = True
        expected_pos = [2, 1]
        expected_vel = [-0.2, 0.5]
        position = [-2, 1]
        velocity = [0.2, 0.5]

        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=velocity)

        np.testing.assert_equal(corr_pos, expected_pos,
                                err_msg="The position correction for the lower bound position was wrong.")
        np.testing.assert_equal(corr_vel, expected_vel,
                                err_msg="The position correction for the lower bound velocity was wrong.")

        expected_pos = [1, 2]
        expected_vel = [0.2, -0.5]
        position = [1, -2]
        velocity = [0.2, 0.5]

        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=velocity)

        np.testing.assert_equal(corr_pos, expected_pos,
                                err_msg="The position correction for the lower bound position was wrong.")
        np.testing.assert_equal(corr_vel, expected_vel,
                                err_msg="The position correction for the lower bound velocity was wrong.")


class periodicBoundaryCondition(unittest.TestCase):
    condition_class = periodicBoundaryCondition
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + condition_class.name, suffix=".obj", dir=tmp_potentials)
    boundary1D = [0, 10]
    boundary2D = [[0, 10], [0, 10]]

    def test_constructor(self):
        print(self.condition_class(boundary=self.boundary1D))

    def test_constructor2D(self):
        print(self.condition_class(boundary=self.boundary2D))

    def test_save_obj_str(self):
        path = self.tmp_out_path
        out_path = self.condition_class(self.boundary2D).save(path=path)
        print(out_path)

    def test_load_str_path(self):
        path = self.tmp_out_path
        out_path = self.condition_class(self.boundary2D).save(path=path)

        cls = self.condition_class.load(path=out_path)
        print(cls)

    def test_apply1D(self):
        cond = self.condition_class(boundary=self.boundary1D)
        cond.verbose = True

        expected_pos = np.array([8])
        expected_vel = np.array([3])
        position = [-2]
        vel = [-3]
        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=vel)

        self.assertEqual(second=corr_pos, first=expected_pos,
                         msg="The position correction for the lower bound position was wrong.")
        self.assertEqual(second=corr_vel, first=expected_vel,
                         msg="The position correction for the lower bound velocity was wrong.")

    def test_apply2D(self):
        cond = self.condition_class(boundary=self.boundary2D)
        cond.verbose = True

        expected_pos = [8, 1]
        expected_vel = np.array([3, 3])
        position = [-2, 1]
        vel = [-3, 3]
        corr_pos, corr_vel = cond.apply(current_position=position, current_velocity=vel)

        np.testing.assert_equal(corr_pos, expected_pos,
                                err_msg="The position correction for the lower bound position was wrong.")
        np.testing.assert_equal(corr_vel, expected_vel,
                                err_msg="The position correction for the lower bound velocity was wrong.")


class positionRestraintCondition(unittest.TestCase):
    condition_class = positionRestraintCondition
    _, tmp_out_path = tempfile.mkstemp(prefix="test_" + condition_class.name, suffix=".obj", dir=tmp_potentials)

    def test_constructor(self):
        print(self.condition_class(position_0=1))

    def test_save_obj_str(self):
        path = self.tmp_out_path
        out_path = self.condition_class(position_0=1).save(path=path)
        print(out_path)

    def test_load_str_path(self):
        path = self.tmp_out_path
        out_path = self.condition_class(position_0=1).save(path=path)

        cls = self.condition_class.load(path=out_path)
        print(cls)

    def test_apply1D(self):
        cond = self.condition_class(position_0=1)
        cond.verbose = True

        expected_pos = [4.5]
        expected_force = [-3]
        position = [-2]

        corr_pos, corr_force = cond.apply(current_position=position)
        self.assertEqual(second=corr_pos, first=expected_pos,
                         msg="The position correction for the lower bound position was wrong.")
        self.assertEqual(second=corr_force, first=expected_force,
                         msg="The position correction for the lower bound force was wrong.")

        expected_pos = [84.5]
        expected_force = [13]
        position = [14]

        corr_pos, corr_force = cond.apply(current_position=position)
        self.assertEqual(second=corr_pos, first=expected_pos,
                         msg="The position correction for the lower bound position was wrong.")
        self.assertEqual(second=corr_force, first=expected_force,
                         msg="The position correction for the lower bound force was wrong.")

    def test_apply2D(self):
        cond = self.condition_class(position_0=1)
        cond.verbose = True
        expected_pos = [4.5, 0]
        expected_force = [-3, 0]
        position = [-2, 1]

        corr_pos, corr_force = cond.apply(current_position=position)
        np.testing.assert_equal(corr_pos, expected_pos,
                                err_msg="The position correction for the lower bound position was wrong.")
        np.testing.assert_equal(corr_force, expected_force,
                                err_msg="The position correction for the lower bound force was wrong.")

        expected_pos = [0, 4.5]
        expected_force = [0, -3]
        position = [1, -2]

        corr_pos, corr_force = cond.apply(current_position=position)
        np.testing.assert_equal(corr_pos, expected_pos,
                                err_msg="The position correction for the lower bound position was wrong.")
        np.testing.assert_equal(corr_force, expected_force,
                                err_msg="The position correction for the lower bound force was wrong.")
