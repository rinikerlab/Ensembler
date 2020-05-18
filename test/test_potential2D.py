import os,sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(__file__+"/.."))

from ensembler.potentials import TwoD

"""
Test Simple 2D Potentials:
"""
class potentialCls_wavePotential(unittest.TestCase):
    def test_constructor(self):
        potential = TwoD.wavePotential()

    def test_energies2D1Pos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = (0,0)
        expected_result = np.array([2])

        potential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        #print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies2DNPos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([2, 1, -1, -1, 2])

        potential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        #print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos2D1Pos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = (0,0)
        expected_result = np.array([0,0], ndmin=2)

        potential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)
        #print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos2DNPos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = [(0, 0), (90, 0), (180, 270), (90, 270), (270, 0), (360, 360)]
        expected_result = np.array([[0,0],[1,0],[0,-1],[1,-1],[-1,0],[0,0]])

        potential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                        y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)

        #print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies),
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

class potentialCls_torsionPotential(unittest.TestCase):
    def test_constructor_SinglePotential(self):
        WavePotential2 = TwoD.wavePotential()
        try:
            torsionPot = TwoD.torsionPotential(wave_potentials=[WavePotential2])
        except:
            return 0
        #print("DID not get an Exception!")
        exit(1)

    def test_constructor_ListPotentials(self):
        WavePotential = TwoD.wavePotential()
        WavePotential2 = TwoD.wavePotential()
        potential = TwoD.torsionPotential(wave_potentials=[WavePotential, WavePotential2])

    def test_energies(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([4, 2,  -2, -2, 4])

        WavePotential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = TwoD.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        potential._set_multiPos_mode()
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies_phase_shifted(self):
        phase_shift1 = (0.0,0)
        phase_shift2 = (180,180)
        multiplicity = (1.0,1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0,0,0)
        radians = False

        positions = [(0,0), (90,90), (180,0), (270,0), (360,0)]
        expected_result = np.array([0, 0, 0, 0, 0])

        WavePotential = TwoD.wavePotential(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        WavePotential2 = TwoD.wavePotential(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude,
                                             y_offset=y_offset, radians=radians)
        potential = TwoD.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        #print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos(self):
        pass
        """
        phase_shift = (0.0,0)
        multiplicity = (1.0,1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0,0,0)
        radians = False

        positions = [(0,0), (90,90), (180,0), (270,0), (360,0)]
        expected_result = np.array([[0], [0], [0], [0], [0]])

        WavePotential = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = TwoD.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = TwoD.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

        """