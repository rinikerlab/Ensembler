import numpy as np
import os
import tempfile
import unittest
from pyemma import msm

from ensembler.analysis.freeEnergyCalculation import  zwanzigEquation, threeStateZwanzigReweighting, bennetAcceptanceRatio
from ensembler.analysis.MarkovStateModel import MarkovStateModel

class test_ZwanzigEquation(unittest.TestCase):
    feCalculation = zwanzigEquation

    def test_constructor(self):
        print(self.feCalculation())

    def test_free_Energy1(self):
        feCalc = self.feCalculation(kT=True)

        # Ensemble Params
        V1_min = 1
        V2_min = 2
        V1_noise = 0.1
        V2_noise = 0.1
        samples = 10000

        V1 = np.random.normal(V1_min, V1_noise, samples)
        V2 = np.random.normal(V2_min, V2_noise, samples)

        dF_ana = V2_min - V1_min
        dF_zwanzig = feCalc.calculate(Vi=V1, Vj=V2)

        np.testing.assert_almost_equal(desired=dF_ana, actual=dF_zwanzig, decimal=2)


class test_BAR(test_ZwanzigEquation):
    feCalculation = bennetAcceptanceRatio

    def test_free_Energy1(self):
        feCalc = self.feCalculation(kT=True)

        # simulate Bar conditions
        samples = 10000

        # ensemble 1
        V1_min = 1
        V1_noise_1 = 0.1

        V2_off = 2
        V2_noise_1 = 0.1

        # ensemble 1
        V1_off = 2
        V1_noise_2 = 0.1
        V2_min = 1
        V2_noise_2 = 0.1

        # Distributions
        V1_1 = np.random.normal(V1_min, V1_noise_1, samples)
        V2_1 = np.random.normal(V2_off, V2_noise_1, samples)

        V1_2 = np.random.normal(V1_off, V1_noise_2, samples)
        V2_2 = np.random.normal(V2_min, V2_noise_2, samples)

        dF_bar = feCalc.calculate(Vi_i=V1_1, Vj_i=V2_1, Vi_j=V1_2, Vj_j=V2_2)

        print(dF_bar)
        dF_ana = 1.000000000000
        np.testing.assert_almost_equal(desired=dF_ana, actual=dF_bar, decimal=2)


class test_BAR(test_ZwanzigEquation):
    feCalculation = bennetAcceptanceRatio

    def test_free_Energy1(self):
        feCalc = self.feCalculation(kT=True)

        # simulate Bar conditions
        samples = 10000

        # ensemble 1
        V1_min = 1
        V1_noise_1 = 0.01
        V2_off = 2
        V2_noise_1 = 0.01

        # ensemble 1
        V1_off = 2
        V1_noise_2 = 0.01
        V2_min = 1
        V2_noise_2 = 0.01

        # Distributions
        V1_1 = np.random.normal(V1_min, V1_noise_1, samples)
        V2_1 = np.random.normal(V2_off, V2_noise_1, samples)

        V1_2 = np.random.normal(V1_off, V1_noise_2, samples)
        V2_2 = np.random.normal(V2_min, V2_noise_2, samples)

        dF_bar = feCalc.calculate(Vi_i=V1_1, Vj_i=V2_1, Vi_j=V1_2, Vj_j=V2_2, verbose=True)

        print(dF_bar)
        dF_ana = 0.000000000000
        np.testing.assert_almost_equal(desired=dF_ana, actual=dF_bar, decimal=2)


class test_threeStateZwanzigReweighting(test_ZwanzigEquation):
    feCalculation = threeStateZwanzigReweighting


    def test_free_Energy1(self):
        feCalc = self.feCalculation(kT=True)

        sample_state1 = 10000
        sample_state2 = 10000

        # State 1 description
        state_1 = 1
        state_1_noise = 0.01

        # State 2 description
        state_2 = 1
        state_2_noise = 0.01

        # OffSampling
        energy_off_state = 10
        noise_off_state = 0.01

        V1 = np.concatenate([np.random.normal(state_1, state_1_noise, sample_state1),
                             np.random.normal(energy_off_state, noise_off_state, sample_state2)])
        V2 = np.concatenate([np.random.normal(energy_off_state, noise_off_state, sample_state1),
                             np.random.normal(state_2, state_2_noise, sample_state2)])
        Vr = np.concatenate([np.random.normal(state_1, state_1_noise, sample_state1),
                             np.random.normal(state_2, state_2_noise, sample_state2)])

        dF_ana = state_2 - state_1
        dFRew_zwanz = feCalc.calculate(Vi=V1, Vj=V2, Vr=Vr)

        np.testing.assert_almost_equal(desired=dF_ana, actual=dFRew_zwanz, decimal=2)


class test_markov_state_model(unittest.TestCase):
    """
    Unittest for analysis/MarkovStateModel.py
    """

    def test_MSM_init(self):

        #test Matrix building
        preprocessed_simulation = [0,0,1]*100
        myMSM = MarkovStateModel(preprocessed_simulation, lag=1)
        expected_result_C = np.array([[100, 100], [99, 0]])
        expected_result_P = np.array([[0.5,0.5],[1,0]])
        np.testing.assert_almost_equal(actual=myMSM.getCountMatrix(), desired=expected_result_C, decimal=1)
        np.testing.assert_almost_equal(actual=myMSM.getTransitionMatrix(), desired=expected_result_P, decimal=.1)

        #test equilibrium distribution, relaxation times and eigenvectors
        expected_result_equil = np.array([0.7,0.3])
        expected_result_relax = np.array([4.5e+15])
        expected_result_eigen = np.array([[0.7,0.3],[-0.5,0.5]])
        np.testing.assert_almost_equal(actual=myMSM.equilibrium(), desired=expected_result_equil, decimal=1)
        #np.testing.assert_almost_equal(actual=myMSM.relaxitionTimes(), desired=expected_result_relax, decimal=1)
        np.testing.assert_almost_equal(actual=myMSM.eigenvectors(), desired=expected_result_eigen, decimal=1)

    def test_MSM_preprocess(self):
    # test extraction from simulation
        X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
        preprocessed_traj = prepareMarkovStateModel(X, n_cluster=2, reweighting=False)
        k_means_relsut = array([1, 1, 1, 0, 0, 0])
        np.testing.assert_almost_equal(actual=preprocessed_traj.clusterKmeans, desired=k_means_relsut, decimal=1)

        # test clustering





