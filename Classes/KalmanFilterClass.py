from pykalman import KalmanFilter
import numpy as np

class KF:
    def __init__(self,
                 transition_matrices = [1],
                 observation_matrices = [1],
                 transition_covariance = 0.01, # Variance ("wiggle room") in how much the true state can change from step to step
                                                # Larger value = allows state to jump more (less smoothing)
                                                # Smaller value = assume underlying state is very stable (more smoothing)
                 observation_covariance = 1.0): # Variance of the measurement noise (how noisy each data point is)
                                                # Larger value = trust observations less (more smoothing)
                                                # Smaller value = trust observations more (less smoothing)

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.set_filter()

    def smooth(self, data: np.ndarray, raw_return=False):
        state_means, state_covariance = self.kf.smooth(data)
        if raw_return:
            return state_means, state_covariance
        else:
            return state_means[:,0]

    def set_filter(self):
        self.kf = KalmanFilter(
            transition_matrices = self.transition_matrices,
            observation_matrices = self.observation_matrices,
            transition_covariance = self.transition_covariance,
            observation_covariance = self.observation_covariance,
            )

    def set_transition_matrices(self, transition_matrices):
        self.transition_matrices = transition_matrices
        self.set_filter()

    def set_observation_matrices(self, observation_matrices):
        self.observation_matrices = observation_matrices
        self.set_filter()

    def set_transition_covariance(self, transition_covariance):
        self.transition_covariance = transition_covariance
        self.set_filter()

    def set_observation_covariance(self, observation_covariance):
        self.observation_covariance = observation_covariance
        self.set_filter()
