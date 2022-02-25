import os
import tempfile
import unittest
from numbers import Number

import numpy as np


"""
TEST for Potential Scaffold:
"""
from ensembler.potentials._basicPotentials import _potentialCls, _potentialNDCls


class test_potentialCls(unittest.TestCase):
    potential_class = _potentialCls
    tmp_test_dir: str = None

    def setUp(self) -> None:
        test_dir = os.getcwd() + "/tests_out"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        if __class__.tmp_test_dir is None:
            __class__.tmp_test_dir = tempfile.mkdtemp(dir=test_dir, prefix="tmp_test_potentials")
        _, self.tmp_out_path = tempfile.mkstemp(prefix="test_" + self.potential_class.name, suffix=".obj", dir=__class__.tmp_test_dir)

    def test_constructor(self):
        potential = self.potential_class()
        print(potential)

    def test_save_obj_str(self):
        path = self.tmp_out_path
        out_path = self.potential_class().save(path=path)
        print(out_path)

    def test_load_str_path(self):
        path = self.tmp_out_path
        out_path = self.potential_class().save(path=path)

        cls = self.potential_class.load(path=out_path)
        print(cls)


class potentialNDCls(unittest.TestCase):
    def test_instantiationError(self):
        try:
            potential = _potentialNDCls()
        except NotImplementedError:
            print("Caught correct Error.")
        except Exception as err:
            raise Exception("Something went wrong here: " + str(err.args))


"""
TEST for Potentials 1D
"""
from ensembler.potentials import OneD


class potentialCls_flatwell(test_potentialCls):
    potential_class = OneD.flatwellPotential

    def test_energies(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [0, 2, 1, 0.5]
        expected_result = np.array([0, 10, 0, 0])

        potential = self.potential_class(x_range=x_range, y_max=y_max, y_min=y_min)

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")

    def test_dVdpos(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [0, 2, 1, 0.5]
        potential = self.potential_class(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([np.inf, 0, np.inf, 0])

        energies = potential.force(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")


class potentialCls_harmonicOsc1D(test_potentialCls):
    potential_class = OneD.harmonicOscillatorPotential

    def test_energies(self):
        fc = 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [0, 2, 1, 0.5]
        expected_result = np.array([0, 2, 0.5, 0.125])

        potential = self.potential_class(k=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")

    def test_dVdpos(self):
        fc: float = 1.0
        x_shift: float = 0.0
        y_shift: float = 0.0
        positions = [0, 0.5, 1, 2]
        expected_result = np.array([0, 0.5, 1, 2])

        potential = self.potential_class(k=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.force(positions)
        # print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")


class potentialCls_wavePotential(test_potentialCls):
    potential_class = OneD.wavePotential

    def test_energies(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dVdpos(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([0, -1, 0, 1, 0])

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        energies = potential.force(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )


class potentialCls_torsionPotential(test_potentialCls):
    potential_class = OneD.torsionPotential

    def test_constructor_SinglePotential(self):
        WavePotential = OneD.wavePotential()
        potential = self.potential_class(wavePotentials=WavePotential)

    def test_constructor_ListPotentials(self):
        WavePotential = OneD.wavePotential()
        WavePotential2 = OneD.wavePotential()
        potential = self.potential_class(wavePotentials=[WavePotential, WavePotential2])

    def test_energies_singlepot(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = OneD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset)
        potential = OneD.torsionPotential(wavePotentials=WavePotential, radians=radians)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies_singlepo_list(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = OneD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        potential = OneD.torsionPotential(wavePotentials=[WavePotential])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([2, 0, -2, 0, 2])

        WavePotential = OneD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset)
        WavePotential2 = OneD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset)
        potential = OneD.torsionPotential(wavePotentials=[WavePotential, WavePotential2], radians=radians)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies_phase_shifted(self):
        phase_shift1 = 0.0
        phase_shift2 = 180
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([0.40153993, 0.80115264, -0.40153993, -0.80115264, 0.40153993])

        WavePotential = OneD.wavePotential(
            phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        WavePotential2 = OneD.wavePotential(
            phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        potential = OneD.torsionPotential(wavePotentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dVdpos(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([0, -2, 0, 2, 0])

        WavePotential = OneD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        WavePotential2 = OneD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_shift=y_offset, radians=radians
        )
        potential = OneD.torsionPotential(wavePotentials=[WavePotential, WavePotential2])
        energies = potential.force(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )


class potentialCls_coulombPotential(test_potentialCls):
    potential_class = OneD.coulombPotential

    def test_energies(self):
        q1 = 1
        q2 = 1
        epsilon = 1

        positions = [0, 0.2, 0.5, 1, 2, 360]
        expected_result = np.array([np.inf, 0.397887358, 0.159154943, 0.0795774715, 0.0397887358, 0.000221048532])

        potential = OneD.coulombPotential(q1=q1, q2=q2, epsilon=epsilon)
        energies = potential.ene(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dVdpos(self):
        q1 = 1
        q2 = 1
        epsilon = 1

        positions = [0, 0.2, 0.5, 1, 2, 360]
        expected_result = np.array([-np.inf, -1.98943679, -0.31830988, -0.0795774715, -0.0198943679, -0.000000614023700])

        potential = OneD.coulombPotential(q1=q1, q2=q2, epsilon=epsilon)
        energies = potential.force(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )


class potentialCls_lennardJonesPotential(test_potentialCls):
    potential_class = OneD.lennardJonesPotential

    def test_energies(self):
        c6: float = 1 ** (-1)
        c12: float = 1 ** (-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array(
            [
                np.nan,
                7.99999200 * 10**12,
                1.95300000 * 10**9,
                3.22560000 * 10**4,
                0.00000000,
                -1.23046875 * 10**-1,
                -1.09588835 * 10**-2,
                -1.71464089 * 10**-4,
            ]
        )
        potential = OneD.lennardJonesPotential(sigma=1, epsilon=2, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )

    def test_dVdpos(self):
        c6: float = 1 ** (-1)
        c12: float = 1 ** (-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array(
            [
                np.nan,
                -9.59999520 * 10**14,
                -1.17183750 * 10**11,
                -7.80288000 * 10**5,
                -4.80000000 * 10,
                3.63281250 * 10**-1,
                2.18876602 * 10**-2,
                1.71460414 * 10**-4,
            ]
        )

        potential = OneD.lennardJonesPotential(sigma=1, epsilon=2, x_shift=x_shift, y_shift=y_shift)
        forces = potential.force(positions)

        self.assertEqual(
            type(expected_result), type(forces), msg="returnType of " + potential.name + " was not correct! it should be an np.array"
        )
        np.testing.assert_allclose(
            desired=expected_result, actual=forces, atol=0.002, err_msg="The results of " + potential.name + " are not correct!"
        )


class potentialCls_lennardJonesForceFieldPotential(test_potentialCls):
    potential_class = OneD.lennardJonesForceFieldPotential

    def test_energies(self):
        c6: float = 1 ** (-1)
        c12: float = 1 ** (-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array([np.nan, 2.4414062e08, 4.0940000e03, 0.0000000e00, -4.9975586e-01, -3.3333145e-01, -1.6666667e-01])

        potential = OneD.lennardJonesForceFieldPotential(c6=c6, c12=c12, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )

    def test_dVdpos(self):
        c6: float = 1 ** (-1)
        c12: float = 1 ** (-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array(
            [
                np.nan,
                -1.20000000 * 10**14,
                -1.46484375 * 10**10,
                -9.83000000 * 10**+4,
                -1.10000000 * 10**1,
                2.48535156 * 10**-1,
                1.11103584 * 10**-1,
                2.77777769 * 10**-2,
            ]
        )

        potential = OneD.lennardJonesForceFieldPotential(c6=c6, c12=c12, x_shift=x_shift, y_shift=y_shift)
        energies = potential.force(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_allclose(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!"
        )


class potentialCls_doubleWellPot1D(test_potentialCls):
    potential_class = OneD.doubleWellPotential

    def test_energies(self):
        Vmax = 100
        a = 0
        b = 8

        positions = np.linspace(-10, 10, num=5)
        expected_result = np.array([31.640625, 37.13378906, 100, 37.13378906, 31.640625])

        potential = self.potential_class(Vmax=Vmax, a=a, b=b)
        energies = potential.ene(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )

    def test_dVdpos(self):
        Vmax = 100
        a = 0
        b = 8

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array([0, -0.62490234, -1.24921875, -3.11279297, -6.15234375, -11.71875, -16.11328125, -16.40625])

        potential = self.potential_class(Vmax=Vmax, a=a, b=b)
        energies = potential.force(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )


class potentialCls_fourWellPot1D(test_potentialCls):
    potential_class = OneD.fourWellPotential

    def test_energies(self):
        Vmax = 100
        a = 0
        b = 8

        positions = np.linspace(-10, 10, num=5)
        expected_result = np.array([10200.0, 2700.0, 200.0, 448.89, 187.26])

        potential = self.potential_class(Vmax=Vmax, a=a, b=b)
        energies = potential.ene(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )

    def test_dVdpos(self):
        Vmax = 100
        a = 0
        b = 8

        positions = [0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array([2.00e01, 4.00e01, 1.00e02, 2.00e02, 4.00e02, 5.943e02, -2.1524e02])

        potential = self.potential_class(Vmax=Vmax, a=a, b=b)
        energies = potential.force(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )


class potentialCls_gaussPotential1D(test_potentialCls):
    potential_class = OneD.gaussPotential

    def test_energies(self):
        Vmax = 100
        a = 0
        b = 8

        positions = np.linspace(-10, 10, num=5)
        expected_result = np.array([1.93e-22, 3.73e-06, 1.00e00, 3.73e-06, 1.93e-22])

        potential = self.potential_class()
        energies = potential.ene(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )

    def test_dVdpos(self):
        Vmax = 100
        a = 0
        b = 8

        positions = [0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array([-9.95e-02, -1.96e-01, -4.41e-01, -6.07e-01, -2.71e-01, -3.33e-02, -9.14e-08])

        potential = self.potential_class()
        energies = potential.force(positions)

        # print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!", decimal=2
        )


class potentialCls_sumPotentials(test_potentialCls):
    potential_class = OneD.sumPotentials
    a = OneD.harmonicOscillatorPotential()
    b = OneD.harmonicOscillatorPotential(x_shift=2)
    c = OneD.harmonicOscillatorPotential(x_shift=-2)

    def test_constructor_ListPotentials(self):
        potential = self.potential_class([self.a, self.b, self.c])

    def test_energies(self):
        positions = np.linspace(-3, 3, 10)
        expected_result = np.array([17.5, 12.16666667, 8.16666667, 5.5, 4.16666667, 4.16666667, 5.5, 8.16666667, 12.16666667, 17.5])

        potential = self.potential_class([self.a, self.b, self.c])
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos(self):
        positions = np.linspace(-3, 3, 10)
        expected_result = np.array([-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0])

        potential = self.potential_class([self.a, self.b, self.c])
        forces = potential.force(positions)

        print(forces)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )


"""
TEST for perturbed Potentials 1D
"""


class potentialCls_perturbedLinearCoupledPotentials(test_potentialCls):
    potential_class = OneD.linearCoupledPotentials

    def test_constructor(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0.5)

    def test_energies(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam1 = 0
        expected_result1 = np.array([12.5, 0, 12.5, 50, 112.5])

        potential.set_lambda(lam=lam1)
        energies = potential.ene(positions)
        # print(energies)
        self.assertEqual(type(expected_result1), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result1,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam1)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies only for pot HB
        lam2 = 1
        expected_result2 = np.array([112.5, 50, 12.5, 0, 12.5])

        potential.set_lambda(lam=lam2)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result2), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result2,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam2)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies merged for pot HB and HA
        lam3 = 0.5
        expected_result3 = np.array([62.5, 25, 12.5, 25, 62.5])

        potential.set_lambda(lam=lam3)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result3), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result3,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam3)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

    def test_dVdpos(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = OneD.linearCoupledPotentials(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lambda(lam=lam)
        expected_result = np.array([-5, 0, 5, 10, 15])

        energies = potential.force(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies only for pot HB
        lam = 1
        expected_result = np.array([-15, -10, -5, 0, 5])

        potential.set_lambda(lam=lam)
        energies = potential.force(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([-10, -5, 0, 5, 10])

        potential.set_lambda(lam=lam)
        energies = potential.force(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

    def test_dHdlam(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lambda(lam=lam)
        expected_result = np.array([100, 50, 0, -50, -100])

        energies = potential.dvdlam(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )


class potentialCls_perturbed_exponentialCoupledPotentials(test_potentialCls):
    potential_class = OneD.exponentialCoupledPotentials

    def test_constructor(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb)

    def test_energies(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        expected_result = np.array([12.5, 0, 12.2, 0, 12.5])

        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=1,
        )

    def test_dVdpos(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(
            Va=ha,
            Vb=hb,
        )

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        expected_result = np.array([-1304235.5118838537, 0.0, 1.9168317203608185e-05, 1.469697537672566e-10, 8.451488578640889e-16])

        energies = potential.force(positions)
        ##print("GOT",  list(energies))
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")


class potentialCls_perturbed_envelopedPotentials(test_potentialCls):
    potential_class = OneD.envelopedPotential

    def test_ene_1Pos(self):
        potential = self.potential_class(s=100)
        positions = 0
        expected_results = 0.0
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

        positions = 3
        expected_results = np.squeeze(np.array([0.0]))
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

        positions = 1.5
        expected_results = np.squeeze(np.array([1.1180685281944005]))
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

        positions = 6
        expected_results = np.squeeze(np.array([4.5]))
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

        positions = -3
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

    def test_ene_NPos(self):
        potential = self.potential_class(s=100)
        positions = (-100, -3, 0, 2, 3, 6, 103)
        expected_results = np.array([5.0e03, 4.5, 0, 0.5, 0, 4.5, 5.0e03])

        actual_energies = potential.ene(positions)
        # CHECK
        self.assertEqual(
            type(expected_results), type(actual_energies), msg="returnType of potential was not correct! it should be an np.array"
        )
        np.testing.assert_almost_equal(
            desired=expected_results,
            actual=actual_energies,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
            decimal=1,
        )

    def test_s_change(self):
        potential = self.potential_class(s=100)
        positions = 1.5
        expected_results = np.squeeze(np.array([1.1180685281944005]))
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

        potential.s_i = 0.01
        expected_results = np.squeeze(np.array([-68.18971805599453]))
        actual_energies = potential.ene(positions)

        # CHECK
        self.assertTrue(
            isinstance(actual_energies, Number),
            msg="returnType of potential was not correct! got actually: " + str(actual_energies) + " expected: " + str(expected_results),
        )
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
        )

    def test_force_1Pos(self):
        potential = self.potential_class(s=100)
        positions = 0
        expected_results = np.squeeze(np.array([0.0]))
        actual_energies = potential.force(positions)
        print("Actual Result1: ", actual_energies)

        # CHECK
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

        positions = 3
        expected_results = np.squeeze(np.array([0.0]))
        actual_energies = potential.force(positions)
        print("Actual Result2: ", actual_energies)

        # CHECK
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

        positions = 1.5
        expected_results = np.squeeze(0.0)
        actual_energies = potential.force(positions)
        print("Actual Result3: ", actual_energies)

        # CHECK
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

        positions = 6
        expected_results = np.squeeze(np.array([-3]))
        actual_energies = potential.force(positions)
        print("Actual Result4: ", actual_energies)

        # CHECK
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

        positions = -3
        expected_results = np.squeeze(np.array([3]))
        actual_energies = potential.force(positions)

        # CHECK
        self.assertAlmostEqual(
            first=expected_results,
            second=actual_energies,
            delta=0.001,
            msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

    def test_force_NPos(self):
        potential = self.potential_class(s=100)
        positions = (-100, -15, -3, 0, 1, 1.5, 2, 3, 6, 18, 103)
        expected_results = np.array([100, 15, 3, 0, -1, 0, 1, 0, -3, -15, -100])
        actual_energies = potential.force(positions)

        # CHECK
        np.testing.assert_almost_equal(
            desired=expected_results,
            actual=actual_energies,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
            decimal=1,
        )


class potentialCls_perturbed_hybridCoupledPotentials(test_potentialCls):
    potential_class = OneD.hybridCoupledPotentials

    def test_constructor(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

    def test_energies(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        expected_result = np.array([12.5, 0, 12.5, 50, 112.5])

        potential.set_lambda(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies only for pot HB
        lam = 1
        expected_result = np.array([112.5, 50, 12.5, 0, 12.5])

        potential.set_lambda(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([13.19, 0.69, 12.5, 0.69, 13.19])

        potential.set_lambda(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

    def test_dVdpos(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lambda(lam=lam)
        expected_result = np.array([-5.00e00, -0.00e00, -0.00e00, 7.87e-54, 5.00e00])

        energies = potential.force(positions)
        ##print("GOT",  list(energies))
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        # np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        # energies only for pot HB
        lam = 1
        expected_result = np.array([-2.6622529026263318e17, -680412108183.5752, -1304235.5118838537, 0.0, 1.9168317203608185e-05])

        potential.set_lambda(lam=lam)
        energies = potential.force(positions)

        ##print("GOT2",  list(energies))
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        # np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([-5.00000000e00, -0.00000000e00, -0.00000000e00, 7.87379318e-54, 5.00000000e00])
        # [-1.331126451319687e+17, -340206054091.7876, -652117.7559323427, 7.34848768836283e-11,
        # 9.584158602226667e-06])

        potential.set_lambda(lam=lam)
        energies = potential.force(positions)

        ##print("GOT3",  list(energies))
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
            decimal=2,
        )

    def test_dHdlam(self):
        ha = OneD.harmonicOscillatorPotential(k=1.0, x_shift=-5.0)
        hb = OneD.harmonicOscillatorPotential(k=1.0, x_shift=5.0)
        potential = self.potential_class(Va=ha, Vb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lambda(lam=lam)
        expected_result = np.array([1.000000e00, 1.000000e00, -0.000000e00, -5.184706e21, -2.688117e43])

        energies = potential.dvdlam(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_allclose(
            desired=expected_result,
            actual=energies,
            err_msg="The results of "
            + potential.name
            + " are not correct wit lambda "
            + str(lam)
            + "!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(energies),
        )


class potentialCls_perturbed_lambdaEnvelopedPotentials(test_potentialCls):
    potential_class = OneD.lambdaEDSPotential

    def test_ene_1Pos(self):
        potential = self.potential_class(s=100)
        positions = [0, 3, 1.5, 6, -3]
        expected_results = np.squeeze(np.array([0.007, 0.007, 1.125, 4.507, 4.507]))
        actual_energies = []

        for position in positions:
            actual_energy = potential.ene(position)
            actual_energies.append(actual_energy)

        # CHECK
        np.testing.assert_almost_equal(
            desired=expected_results, actual=actual_energies, decimal=3, err_msg="The results of " + potential.name + " are not correct!\n"
        )

    def test_ene_NPos(self):
        potential = self.potential_class(s=100)
        positions = (-100, -3, 0, 2, 3, 6, 103)
        expected_results = np.array([5.0e03, 4.5e00, 6.9e-03, 5.1e-01, 6.9e-03, 4.5e00, 5.0e03])

        actual_energies = potential.ene(positions)
        print(actual_energies)
        # CHECK
        self.assertEqual(
            type(expected_results), type(actual_energies), msg="returnType of potential was not correct! it should be an np.array"
        )
        np.testing.assert_almost_equal(
            desired=expected_results,
            actual=actual_energies,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
            decimal=1,
        )

    def test_s_change(self):
        potential = self.potential_class(s=100)
        positions = [1.5, 0.01]
        expected_results = np.squeeze(np.array([1.125, 0.007]))
        actual_energies = []
        for position in positions:
            actual_energy = potential.ene(position)
            actual_energies.append(actual_energy)

        # CHECK
        np.testing.assert_almost_equal(
            desired=expected_results, actual=actual_energies, decimal=3, err_msg="The results of " + potential.name + " are not correct!\n"
        )

    def test_force_1Pos(self):
        potential = self.potential_class(s=100)
        positions = [0, 3, 1.5, 6, -3]
        expected_results = np.array([1.108e-195, -1.108e-195, 0.000e000, -3.000e000, 3.000e000])

        actual_energies = []
        for position in positions:
            actual_energy = potential.force(position)
            actual_energies.append(actual_energy)
        print(actual_energies)
        # CHECK
        np.testing.assert_almost_equal(
            desired=expected_results,
            actual=actual_energies,
            decimal=3,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tForces: "
            + str(actual_energies),
        )

    def test_force_NPos(self):
        potential = self.potential_class(s=100)
        positions = (-100, -15, -3, 0, 1, 1.5, 2, 3, 6, 18, 103)
        expected_results = np.array(
            [1.0e002, 1.5e001, 3.0e000, 1.1e-195, -1.0e000, 0.0e000, 1.0e000, -1.1e-195, -3.0e000, -1.5e001, -1.0e002]
        )
        actual_energies = potential.force(positions)
        print(actual_energies)
        # CHECK
        np.testing.assert_almost_equal(
            desired=expected_results,
            actual=actual_energies,
            err_msg="The results of "
            + potential.name
            + " are not correct!\n\tPositions: "
            + str(positions)
            + "\n\tEnergies: "
            + str(actual_energies),
            decimal=1,
        )


"""
Test Simple 2D Potentials:
"""
from ensembler.potentials import TwoD


class potentialCls_2D_harmonicOscillatorPotential(test_potentialCls):
    potential_class = TwoD.harmonicOscillatorPotential

    def test_energies2DNPos(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 1.0])

        potential = self.potential_class()
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos2DNPos(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0]])

        potential = self.potential_class()
        forces = potential.force(positions)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )

        # for ind, (expected, actual) in enumerate(zip(expected_result, forces.T)):


class potentialCls_2D_wavePotential(test_potentialCls):
    potential_class = TwoD.wavePotential

    def test_energies2D1Pos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = np.array([0.0, 0.0])
        expected_result = np.array([2])

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=np.array(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies2DNPos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = np.array([[0.0, 0.0], [90.0, 0.0], [180.0, 270.0], [270.0, 180.0], [360.0, 360.0]])
        expected_result = np.array([2, 1, -1, -1, 2])

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos2D1Pos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False
        positions = np.array([0, 0])
        expected_result = np.array([0, 0], ndmin=1)

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        forces = potential.force(positions)
        # print(forces)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )

    def test_dHdpos2DNPos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = np.array([[0, 0], [90, 0], [180, 270], [90, 270], [270, 0], [360, 360]])
        expected_result = np.array([[0, 0], [-1, 0], [0, 1], [-1, 1], [1, 0], [0, 0]])

        potential = self.potential_class(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        forces = potential.force(positions)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")

        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )


class potentialCls_2D_torsionPotential(test_potentialCls):
    potential_class = TwoD.addedWavePotential

    def test_constructor_ListPotentials(self):
        WavePotential = TwoD.wavePotential()
        WavePotential2 = TwoD.wavePotential()
        potential = self.potential_class(wave_potentials=[WavePotential, WavePotential2])

    def test_energies(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = False

        positions = np.array([(0, 0), (90, 0), (180, 270), (270, 180), (360, 360)])
        expected_result = np.array([4, 2, -2, -2, 4])

        WavePotential = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        WavePotential2 = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        potential = self.potential_class(wave_potentials=[WavePotential, WavePotential2])

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies_phase_shifted(self):
        phase_shift1 = (0.0, 0)
        phase_shift2 = (180, 180)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)

        positions = np.array([(0, 0), (90, 90), (180, 0), (270, 0), (360, 0)])
        expected_result = np.array([0.00000000e00, -7.35035672e-15, 0.00000000e00, 3.66373598e-15, 0.00000000e00])

        WavePotential = TwoD.wavePotential(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset)
        WavePotential2 = TwoD.wavePotential(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset)
        potential = self.potential_class(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = False

        positions = np.array([(0, 0), (90, 90), (180, 0), (270, 0), (360, 0)])
        expected_result = np.array(
            [
                [0.0000000e00, 0.0000000e00],
                [-2.0000000e00, -2.0000000e00],
                [-2.4492936e-16, 0.0000000e00],
                [2.0000000e00, 0.0000000e00],
                [4.8985872e-16, 0.0000000e00],
            ]
        )

        WavePotential = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        WavePotential2 = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        potential = self.potential_class(wave_potentials=[WavePotential, WavePotential2])
        forces = potential.force(positions)

        print(forces)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )


class potentialCls_2D_sumPotentials(test_potentialCls):
    potential_class = TwoD.sumPotentials

    def test_constructor_ListPotentials(self):
        WavePotential = TwoD.wavePotential()
        WavePotential2 = TwoD.wavePotential()
        potential = self.potential_class(potentials=[WavePotential, WavePotential2])

    def test_energies(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = True

        positions = np.deg2rad(np.array([(0, 0), (90, 0), (180, 270), (270, 180), (360, 360)]))

        expected_result = np.array([4, 2, -2, -2, 4])

        WavePotential = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        WavePotential2 = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        potential = self.potential_class(potentials=[WavePotential, WavePotential2])

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_energies_phase_shifted(self):
        phase_shift1 = (0.0, 0)
        phase_shift2 = (180, 180)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)

        positions = np.array([(0, 0), (90, 90), (180, 0), (270, 0), (360, 0)])
        expected_result = np.array([0.00000000e00, -7.35035672e-15, 0.00000000e00, 3.66373598e-15, 0.00000000e00])

        WavePotential = TwoD.wavePotential(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset)
        WavePotential2 = TwoD.wavePotential(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset)
        potential = self.potential_class(potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = True

        positions = np.deg2rad(np.array([(0, 0), (90, 90), (180, 0), (270, 0), (360, 0)]))
        expected_result = np.array(
            [
                [0.0000000e00, 0.0000000e00],
                [-2.0000000e00, -2.0000000e00],
                [-2.4492936e-16, 0.0000000e00],
                [2.0000000e00, 0.0000000e00],
                [4.8985872e-16, 0.0000000e00],
            ]
        )

        WavePotential = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        WavePotential2 = TwoD.wavePotential(
            phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians
        )
        potential = self.potential_class(potentials=[WavePotential, WavePotential2])
        forces = potential.force(positions)

        print(forces)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )


"""
Test Simple ND Potentials:
"""
from ensembler.potentials import ND


class potentialCls_ND_harmonicOscillatorPotential(test_potentialCls):
    potential_class = ND.harmonicOscillatorPotential

    def test_energies3DNPos(self):
        positions = np.array([[0, 0, 0], [1, 0, 1], [-1, 0, 1], [0, 1, 0], [0, -1, -1], [-1, -1, 1]])
        expected_result = np.array([0.0, 1, 1, 0.5, 1, 1.5])

        potential = self.potential_class(nDimensions=3)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos3DNPos(self):
        positions = np.array([[0, 0, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 0], [0, -1, 0], [-1, -1, -1]])
        expected_result = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, -1.0, -1.0]]
        )

        potential = self.potential_class(nDimensions=3)
        forces = potential.force(positions)
        print(str(forces))
        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )

        # for ind, (expected, actual) in enumerate(zip(expected_result, forces.T)):


class potentialCls_ND_sumPotentials(test_potentialCls):
    potential_class = TwoD.sumPotentials

    def test_constructor_ListPotentials(self):
        potential = self.potential_class()

    def test_energies(self):
        positions = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 2, 0)])
        expected_result = np.array([1.5, 1.5, 1.5, 1.5, 5.5])

        potential = self.potential_class()

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=list(expected_result),
            actual=list(energies),
            err_msg="The results of " + potential.name + " are not correct!",
            decimal=8,
        )

    def test_dHdpos(self):
        positions = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 2, 0)])
        expected_result = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [3.0, 3.0, -1.0]])

        potential = self.potential_class()
        forces = potential.force(positions)

        print(forces)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!", decimal=8
        )


"""
biased potentials
"""


class potentialCls_addedPotentials(test_potentialCls):
    potential_class = OneD.addedPotentials

    def test_energies(self):
        positions = [0, 2, 1, 0.5]
        expected_result = np.array([1.0, 2.135335283236613, 1.1065306597126334, 1.0074969025845955])

        potential = self.potential_class()
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")

    def test_dVdpos(self):
        positions = [0, 0.5, 1, 2]
        expected_result = np.array([0.0, 0.05875154870770227, 0.3934693402873666, 1.7293294335267746])

        potential = self.potential_class()

        energies = potential.force(positions)
        # print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")


class potentialCls_metadynamics(test_potentialCls):
    potential_class = OneD.metadynamicsPotential

    def test_energies(self):
        positions = [0, 2, 1, 0.5]
        expected_result = np.array([0, 2, 0.5, 0.125])

        potential = self.potential_class()
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of " + potential.name + " are not correct!")

    def test_dVdpos_NPos(self):
        positions = np.array([0, 0.5, 1, 2])
        expected_result = np.array([0, 0.5, 1, 2])

        potential = self.potential_class()

        forces = potential.force(positions)
        # print(energies)

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!"
        )

    def test_dVdpos_1Pos(self):
        positions = 0
        expected_result = 0

        potential = self.potential_class()

        forces = potential.force(positions)
        # print(energies)

        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!"
        )


class potentialCls_addedPotentials2D(test_potentialCls):
    potential_class = TwoD.addedPotentials

    def test_energies(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.array([1.0, 1.1065307, 1.1065307, 1.1065307, 1.1065307, 1.3678794])

        potential = self.potential_class()
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!"
        )

    def test_dVdpos(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.array(
            [[0.0, 0.0], [0.39346934, 0.0], [-0.39346934, 0.0], [0.0, 0.39346934], [0.0, -0.39346934], [-0.63212056, -0.63212056]]
        )

        potential = self.potential_class()

        forces = potential.force(positions)

        print(str(forces))

        self.assertEqual(type(expected_result), type(forces), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!"
        )


class potentialCls_metadynamics2D(test_potentialCls):
    potential_class = TwoD.metadynamicsPotential

    def test_energies(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 1.0])

        potential = self.potential_class()
        energies = potential.ene(positions)

        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!"
        )

    def test_energies(self):
        positions = np.array([0, 0])
        expected_result = np.array([0.0])

        potential = self.potential_class()
        energies = potential.ene(positions)

        np.testing.assert_almost_equal(
            desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!"
        )

    def test_dVdpos(self):
        positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1]])
        expected_result = np.squeeze(np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0]]))

        potential = self.potential_class()
        forces = potential.force(positions)
        print(forces.shape, forces)
        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!"
        )

    def test_dVdpos_1Pos(self):
        positions = [0, 0]
        expected_result = [0, 0]

        potential = self.potential_class()

        forces = potential.force(positions)
        # print(energies)

        np.testing.assert_almost_equal(
            desired=expected_result, actual=forces, err_msg="The results of " + potential.name + " are not correct!"
        )


if __name__ == "__main__":
    unittest.main()
