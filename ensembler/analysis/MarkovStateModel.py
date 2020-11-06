'''
This class provides functionalities to build a kinetic Markov State Model. If the enhanced sampling was used for the
simulation, this class provides different reweighting algorithms.
'''
import numpy as np
from pyemma import msm
from sklearn.cluster import KMeans


class MarkovStateModel():

    def __init__(self, d_traj, lag=1):
        '''
        Creates Markov State Matrix based on provided pre-processed simulation data
        Returns
        -------
        '''
        self.lag = lag  # lagtime
        self.d_traj = d_traj
        #get number of unique items in traj
        nstates = np.max(self.d_traj)+1 #+1 is needed because if max state is n due to python numbering we have n+1 states
        #create empty matrix
        self.C = np.zeros((nstates, nstates))

        # build count matrix without reweighting
        l = len(self.d_traj)
        # start in state i go to state j
        for i in range(l - self.lag):
            j = i + self.lag
            self.C[int(self.d_traj[i]), int(self.d_traj[j])] = self.C[int(d_traj[i]), int(d_traj[j])] + 1

        # normalize Matrix
        self.normalizeMatrix()

    def getCountMatrix(self):
         '''
         Returns the Count Matrix of the MSM
         Returns
         -------

         '''
         return self.C

    def getTransitionMatrix(self):
         '''
         Returns the Transition Matrix of the MSM
         Returns
         -------

         '''
         return self.P

    def normalizeMatrix(self):
         '''
         normalizes count Matrix of the MSM
         Returns
         -------
         '''
         #Normalize counts matrix
         self.P = np.zeros((len(self.C), len(self.C)))
         for i in range(len(self.C)):
             if np.sum(self.C[i,:]) > 0:
                 self.P[i,:] = self.C[i,:]/np.sum(self.C[i,:])

    def relaxitionTimes(self):
        '''
        Uses the Markov State Matrix and returns the relaxation times in decendent order unit [steps]
        Returns
        -------
        '''
        mm = msm.markov_model(self.P)
        eigenvalues = mm.eigenvalues()
        self.relax = -self.lag / np.log(eigenvalues[1:])
        return self.relax

    def eigenvectors(self):
        '''
        Uses the Markov State Matrix and returns the eigenvectors (description of the relaxiation states) in
        decendent order
        Returns
        -------

        '''
        mm = msm.markov_model(self.P)
        self.eigenvectors = mm.eigenvectors_left() #check if left or right eigenvector is needed
        return self.eigenvectors

    def equilibrium(self):
        '''
        Uses the Markov State Matrix and returns the equilibrium distribution
        Returns
        -------

        '''
        mm = msm.markov_model(self.P)
        self.equilib = mm.stationary_distribution
        return self.equilib


class prepareMarkovStateModel():

    def __init__(self, system, n_cluster= None, reweighting=False):
        '''
        Provides pre-processed simulation data
        Returns
        -------
        '''
        self.reweighting = reweighting
        self.n_cluster = n_cluster
        self.system = system
        self.traj = system.trajectory
        #get number of unique items in traj
        nstates = np.max(self.traj)+1 #+1 is needed because if max state is n due to python numbering we have n+1 states

    def prepareSimulation(self):
        '''
        Reads the necessary information from the Ensebler simulation object.
        Returns
        -------
        '''

        self.positions = self.traj['position']
        self.temperature = self.traj['temperature']
        if self.reweighting:
            #leave out last energy and force information because they always contribute to step n+1
            #actual - original
            self.energy_diff = self.traj['total_potential_energy'][:-1]-self.traj['potential_energy_orig'][:-1]
            self.force_diff = self.traj['dhdpos'][:-1]-self.traj['dhdpos_orig'][:-1]
            #leave out first random number
            self.random_number = self.traj['previous_random_number'][1:]

    def clusterKmeans(self):
        '''
        Cluster the simulation with a simple k-means clustering algorithm
        Returns
        -------
        '''

        #reshape if dimentionality =1
        if self.positions[0].shape==():
            kmeans = KMeans(n_clusters=self.n_cluster).fit(self.positions.values.reshape(-1, 1))
        else:
            kmeans = KMeans(n_clusters=self.n_cluster).fit(self.positions)

        self.clustercenter = kmeans.cluster_centers_
        self.d_traj = kmeans.labels_
        return self.d_traj

class Reweighting(MarkovStateModel):

    def __init__(self, prepared_traj_obj, tlag):
        '''
        Provides pre-processed simulation data
        Returns
        -------
        '''
        self.traj = prepared_traj_obj
        self.lag = tlag
        self.d_traj = prepared_traj_obj.d_traj

        self.k_B = 1  # everything is in units of k_B

        # build reweighted count matrix
        nstates = np.max(self.d_traj) + 1  # +1 is needed because if max state is n due to python numbering we have n+1 states
        # create empty matrix
        self.C = np.zeros((nstates, nstates))

    def DHAM(self):
        '''
        Performs DHAM reweighting. This reweighting form requires the energy difference between the original and biased
        simulation
        Returns
        -------
        '''

        # build count matrix with reweighting
        l = len(self.d_traj) -1 #remove last state
        for i in range(l - self.lag):
            T = self.traj.temperature[i]
            j = i + self.lag
            DHAM_factor = np.exp((self.traj.energy_diff[j]-self.traj.energy_diff[i])/ (2 * self.k_B * T))
            self.C[int(self.d_traj[i]), int(self.d_traj[j])] = \
                self.C[int(self.d_traj[i]), int(self.d_traj[j])] + DHAM_factor

        # normalize Matrix
        self.normalizeMatrix()


    def Weber_Pande(self):

        '''
        Performs Weber_Pande reweighting. This reweighting form requires the force difference between the original
        and biased simulation and the random number
        Returns
        -------
        '''

        # build count matrix with reweighting

        random_unbiased = self.traj.force_diff.values + self.traj.random_number.values

        T = self.traj.system.temperature # temperature
        gamma = self.traj.system.sampler.gamma # friction
        dt = self.traj.system.sampler.dt  # simulation step
        sigma_square = 2 * self.k_B * T * gamma * (1. / dt)

        # calculate the action difference
        self.A_diff = (self.traj.random_number.values ** 2 - random_unbiased ** 2) / (2 * sigma_square)

        l = len(self.d_traj) -1 #remove last state
        for i in range(l - self.lag):
            j = i + self.lag
            Weber_Pande_factor = np.exp(np.sum(self.A_diff[i:j]))
            self.C[int(self.d_traj[i]), int(self.d_traj[j])] = \
                self.C[int(self.d_traj[i]), int(self.d_traj[j])] + Weber_Pande_factor

        # normalize Matrix
        self.normalizeMatrix()

    def Girsanov(self):
        '''
        Performs Girsanov Reweighting. Needs force differences, energies and random number

        Returns
        -------

        '''
        # build count matrix with reweighting

        T = self.traj.system.temperature  # temperature
        gamma = self.traj.system.sampler.gamma  # friction
        dt = self.traj.system.sampler.dt  # simulation step
        sigma_square = 2 * self.k_B * T * gamma * (1. / dt)

        # calculate startposition reweighting
        self.g_gir = np.exp((-1. / (T * self.k_B)) * self.traj.energy_diff)
        # calculate path reweighting
        # get factors of integral 1
        self.M_gir_I1 = (- self.traj.force_diff.values / np.sqrt(2 * self.k_B * T * gamma)) * self.traj.random_number.values * np.sqrt(dt)
        # get factors of integral 2
        self.M_gir_I2 = ((-self.traj.force_diff.values / np.sqrt(2 * self.k_B * T * gamma)) ** 2) * dt

        l = len(self.d_traj) - 1  # remove last state
        for i in range(l - self.lag):
            j = i + self.lag
            w_gir = np.exp(np.sum(self.M_gir_I1[i:j] - 0.5 * self.M_gir_I2[i:j]))
            Girsanov_factor = w_gir * self.g_gir[i]
            self.C[int(self.d_traj[i]), int(self.d_traj[j])] = \
                self.C[int(self.d_traj[i]), int(self.d_traj[j])] + Girsanov_factor

        # normalize Matrix
        self.normalizeMatrix()

    def pathDHAM(self, integrated=True):

        '''
        Performs pathDHAM reweighting. This reweighting form requires the force difference between the original
        and biased simulation and the random number
        Returns
        -------
        '''

        # build count matrix with reweighting
        T = self.traj.system.temperature # temperature
        gamma = self.traj.system.sampler.gamma # friction
        dt = self.traj.system.sampler.dt  # simulation step
        sigma_square = 2 * self.k_B * T * gamma * (1. / dt)

        # calculate the correction factor
        self.A_corr = dt * (self.traj.traj.dhdpos_orig ** 2 - self.traj.traj.dhdpos ** 2) / (2 * sigma_square)

        # force_factor
        force_fact = (self.traj.force_diff.values / (2 * self.k_B * T)) * \
                     (self.traj.traj.position[1:].values - self.traj.traj.position[:-1].values)

        l = len(self.d_traj) -1 #remove last state
        for i in range(l - self.lag):
            j = i + self.lag
            if integrated:
                factor_1 = np.exp((self.traj.energy_diff[j]-self.traj.energy_diff[i])/ (2 * self.k_B * T))
            else:
                factor_1 = np.exp(-np.sum(force_fact[i:j]))
            pathDHAM_factor =  factor_1 * np.exp(-np.sum(self.A_corr[i:j]))
            self.C[int(self.d_traj[i]), int(self.d_traj[j])] = \
                self.C[int(self.d_traj[i]), int(self.d_traj[j])] + pathDHAM_factor

        # normalize Matrix
        self.normalizeMatrix()


