"""
Augmented GP-EnKF model
"""

import numpy as np
from gpenkf.gp_util.squared_exponential import SquaredExponential


class AugmentedGPEnKF:
    """
    Augmented GP-EnKF model. Has joint ensemble for parameters and state.

    :param parameters: The :class:`~gpenkf.core.parameters.parameters` parameters
    :param learn_gp_parameters: indicator if the GP hyperparameters should be learnt
    :param learn_sigma: indicator if the noise variance should be learnt
    """

    def __init__(self, parameters, learn_gp_parameters=True, learn_sigma=False):

        self.learn_gp_parameters = learn_gp_parameters
        self.parameters = parameters

        if self.learn_gp_parameters:
            self.params_dimensionality = self.parameters.hyperparams_dimensionality
        else:
            self.params_dimensionality = 0

        self.learn_sigma = learn_sigma
        if self.learn_sigma:
            self.params_dimensionality += 1

        self.ensemble_size = self.parameters.ensemble_size
        self.sample_size = self.parameters.sample_size
        self.sigma_y = self.parameters.sigma_y
        self.sigma_eta = self.parameters.sigma_eta
        self.inducing_points_locations = self.parameters.inducing_points_locations
        self.initial_log_gp_params = self.parameters.initial_log_gp_params
        self.initial_log_sigma = self.parameters.initial_log_sigma
        self.min_exp = -20

        self.grid_size = self.parameters.grid_size

        self.augmented_state_size = self.grid_size + self.params_dimensionality

        augmented_state_mean = np.ones((self.grid_size+self.params_dimensionality,))
        augmented_state_cov = self.parameters.init_cov * np.eye(self.augmented_state_size)

        self.augmented_state_ensemble = np.random.multivariate_normal(mean=augmented_state_mean,
                                                                      cov=augmented_state_cov,
                                                                      size=self.ensemble_size)

        self.predictions = np.zeros((self.ensemble_size, self.sample_size))

        self.y_trajectories = np.zeros((self.ensemble_size, self.sample_size))

    def run_iteration(self, x_new, f_new_noisy):
        """
        Perform one iteration of predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """
        self.predict(x_new, f_new_noisy)
        self.update_augmented_state()

    def predict(self, x_new, f_new_noisy):
        """
        Predict step of the predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """
        self.__predict_observations(x_new)
        self.__compute_noisy_trajectories(f_new_noisy)

    def update_augmented_state(self):
        """
        Update step of the predict-update loop.
        """
        sigma_g_y = self.__compute_cross_covariance_of_state_and_prediction()
        sigma_y_y = self.__compute_forecast_error_covariance_matrix_of_predictions()
        kalman_gain_g = self.__compute_kalman_gain_for_state(sigma_g_y, sigma_y_y)
        self.__update_state_internal(kalman_gain_g)

    def __predict_observations(self, x_new):
        for ens_idx in range(self.ensemble_size):

            if self.learn_gp_parameters and self.learn_sigma:
                state, log_hyperparameters, log_sigma = self.__decompose_augmented_state(
                    self.augmented_state_ensemble[ens_idx])
                cov_func = SquaredExponential.from_parameters_vector(np.exp(log_hyperparameters))
                mean, _ = cov_func.predict(x_new, self.inducing_points_locations, np.exp(log_sigma), state)
            elif self.learn_gp_parameters:
                state, log_hyperparameters = self.__decompose_augmented_state(self.augmented_state_ensemble[ens_idx])
                cov_func = SquaredExponential.from_parameters_vector(np.exp(log_hyperparameters))
                mean, _ = cov_func.predict(x_new, self.inducing_points_locations, np.exp(self.initial_log_sigma), state)
            elif self.learn_sigma:
                state, log_sigma = self.__decompose_augmented_state(self.augmented_state_ensemble[ens_idx])
                cov_func = SquaredExponential.from_parameters_vector(np.exp(self.initial_log_gp_params))
                mean, _ = cov_func.predict(x_new, self.inducing_points_locations, np.exp(log_sigma), state)
            else:
                state = self.__decompose_augmented_state(self.augmented_state_ensemble[ens_idx])
                cov_func = SquaredExponential.from_parameters_vector(np.exp(self.initial_log_gp_params))
                mean, _ = cov_func.predict(x_new, self.inducing_points_locations, np.exp(self.initial_log_sigma), state)

            self.predictions[ens_idx] = mean

    def __compute_noisy_trajectories(self, f_new_noisy):
        for ens_idx in range(self.ensemble_size):
            self.y_trajectories[ens_idx] = (
                    f_new_noisy + np.random.normal(loc=0., scale=self.sigma_y, size=(self.sample_size,))).T

    def __compute_cross_covariance_of_state_and_prediction(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        augmented_ensemble_mean = np.mean(self.augmented_state_ensemble, axis=0)
        sigma_as_y = np.zeros((self.augmented_state_size, self.sample_size))
        for i in range(self.ensemble_size):
            sigma_as_y += np.outer(self.augmented_state_ensemble[i] - augmented_ensemble_mean,
                                   self.predictions[i] - predictions_mean)
        sigma_as_y /= (self.ensemble_size - 1)

        return sigma_as_y

    def __compute_forecast_error_covariance_matrix_of_predictions(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        sigma_y_y = np.zeros((self.sample_size, self.sample_size))
        for i in range(self.ensemble_size):
            sigma_y_y += np.outer(self.predictions[i] - predictions_mean, self.predictions[i] - predictions_mean)
        sigma_y_y /= (self.ensemble_size - 1)

        return sigma_y_y

    def __compute_kalman_gain_for_state(self, sigma_g_y, sigma_y_y):
        kalman_gain_as = np.matmul(sigma_g_y, np.linalg.inv(sigma_y_y + self.sigma_y * np.eye(self.sample_size)))

        return kalman_gain_as

    def __update_state_internal(self, kalman_gain_as):
        for i in range(self.ensemble_size):
            self.augmented_state_ensemble[i] = self.augmented_state_ensemble[i] + \
                                               np.matmul(kalman_gain_as, self.y_trajectories[i] - self.predictions[i])
        self.__check_augmented_state_ensemble()

    def __check_augmented_state_ensemble(self):
        if not np.all(np.isfinite(self.augmented_state_ensemble)):
            raise ValueError('g are not finite')

    def __decompose_augmented_state(self, ensemble_member):
        if self.learn_gp_parameters and self.learn_sigma:
            state, log_hyperparameters, log_sigma = np.split(ensemble_member,
                                                             np.cumsum([self.grid_size, self.params_dimensionality-1]))
            return state, log_hyperparameters, log_sigma

        if self.learn_gp_parameters:
            state, log_hyperparameters = np.split(ensemble_member,
                                                  np.cumsum([self.grid_size]))
            return state, log_hyperparameters

        if self.learn_sigma:
            state, log_sigma = np.split(ensemble_member, np.cumsum([self.grid_size]))
            return state, log_sigma

        state = ensemble_member
        return state

    def get_eta_ensemble(self):
        """
        Extract part of the joint ensemble that corresponds to the GP hyperparameters.

        :return: Ensemble of logarithms of GP log hyperparameters
        """
        if self.learn_gp_parameters and self.learn_sigma:
            return self.augmented_state_ensemble[:, self.grid_size:]

        if self.learn_gp_parameters:
            return self.augmented_state_ensemble[:, self.grid_size:]

        if self.learn_sigma:
            return self.augmented_state_ensemble[:, self.grid_size:]

        return None

    def get_g_ensemble(self):
        """
        Extract part of the joint ensemble that corresponds to the noise variance.

        :return: Ensemble of logarithms of noise variance parameters
        """
        return self.augmented_state_ensemble[:, :self.grid_size]

    def get_log_mean_params(self):
        """
        :return: logarithm of mean GP hyperparameters, logarithm of mean noise variance
        """
        if self.learn_gp_parameters and self.learn_sigma:
            _, log_hyperparameters, log_sigma = self.__decompose_augmented_state(
                self.augmented_state_ensemble.mean(axis=0))

        elif self.learn_gp_parameters:
            _, log_hyperparameters = self.__decompose_augmented_state(
                self.augmented_state_ensemble.mean(axis=0))
            log_sigma = self.initial_log_sigma

        elif self.learn_sigma:
            _, log_sigma = self.__decompose_augmented_state(
                self.augmented_state_ensemble.mean(axis=0))
            log_hyperparameters = self.initial_log_gp_params

        else:
            log_hyperparameters, log_sigma = self.initial_log_gp_params, self.initial_log_sigma

        return log_hyperparameters, log_sigma

    def get_g_mean(self):
        """
        :return: mean of state ensemble
        """
        return np.mean(self.get_g_ensemble(), axis=0)

    def compute_nmse(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: NMSE between predicted and true values at sample points
        """
        log_hyperparameters, log_sigma = self.get_log_mean_params()
        g_mean = self.get_g_mean()

        cov_func = SquaredExponential.from_parameters_vector(np.exp(log_hyperparameters))
        mean, _ = cov_func.predict(x_sample, self.inducing_points_locations, np.exp(log_sigma), g_mean)

        return np.mean(np.sqrt((mean-f_true_sample)**2)/np.sqrt(f_true_sample**2))

    def compute_log_likelihood(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: log likelihood of true values at sample points given the estimated model
        """
        log_gp_params, log_sigma = self.get_log_mean_params()
        cov_func = SquaredExponential.from_parameters_vector(np.exp(log_gp_params))
        return cov_func.log_likelihood(f_true_sample, x_sample, np.exp(log_sigma))
