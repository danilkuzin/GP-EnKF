"""
Dual GP-EnKF model
"""

import numpy as np
from gpenkf.gp_util.squared_exponential import SquaredExponential


class DualGPEnKF:
    """
    Dual GP-EnKF model. Has separate ensembles for parameters and state.

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
        self.sigma_eta = self.parameters.sigma_eta * np.eye(self.params_dimensionality, dtype=np.double)
        self.inducing_points_locations = self.parameters.inducing_points_locations
        self.initial_log_gp_params = self.parameters.initial_log_gp_params
        self.initial_log_sigma = self.parameters.initial_log_sigma
        self.min_exp = -20

        if self.learn_gp_parameters and self.learn_sigma:
            params_mean = np.array([self.initial_log_gp_params[0], self.initial_log_gp_params[1], self.initial_log_sigma])
            params_cov = self.parameters.init_cov * np.eye(self.params_dimensionality)
            self.params_ensemble = np.random.multivariate_normal(mean=params_mean,
                                                                 cov=params_cov,
                                                                 size=self.ensemble_size)
        elif self.learn_gp_parameters:
            params_mean = np.array(self.initial_log_gp_params)
            params_cov = self.parameters.init_cov * np.eye(self.params_dimensionality)
            self.params_ensemble = np.random.multivariate_normal(mean=params_mean,
                                                                 cov=params_cov,
                                                                 size=self.ensemble_size)

        elif self.learn_sigma:
            params_mean = np.array(
                [self.initial_log_sigma])
            params_cov = self.parameters.init_cov * np.eye(self.params_dimensionality)
            self.params_ensemble = np.random.multivariate_normal(mean=params_mean,
                                                                 cov=params_cov,
                                                                 size=self.ensemble_size)

        self.grid_size = self.parameters.grid_size
        g_mean = np.zeros((self.grid_size,))
        g_cov = np.eye(self.grid_size)
        self.g_ensemble = np.random.multivariate_normal(mean=g_mean, cov=g_cov, size=self.ensemble_size)

        self.predictions = np.zeros((self.ensemble_size, self.sample_size))

        self.y_trajectories = np.zeros((self.ensemble_size, self.sample_size))

    def run_iteration(self, x_new, f_new_noisy):
        """
        Perform one iteration of predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """

        self.predict(x_new, f_new_noisy)
        if self.params_dimensionality > 0:
            self.update_parameters()
        self.update_state(x_new)

    def predict(self, x_new, f_new_noisy):
        """
        Predict step of the predict-update loop.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """
        self.__sample_parameters()
        self.__predict_observations(x_new)
        self.__compute_noisy_trajectories(f_new_noisy)

    def update_parameters(self):
        """
        Update parameters step of the predict-update loop.
        """
        sigma_eta_y = self.__compute_cross_covariance_of_parameters_and_predictions()
        sigma_y_y = self.__compute_forecast_error_covariance_matrix_of_predictions()
        kalman_gain_eta = self.__compute_kalman_gain_for_parameters(sigma_eta_y, sigma_y_y)
        self.__update_parameters_internal(kalman_gain_eta)

    def update_state(self, x_new):
        """
        Update state step of the predict-update loop.
        """
        self.__predict_observations(x_new)
        sigma_g_y = self.__compute_cross_covariance_of_state_and_prediction()
        sigma_y_y = self.__compute_forecast_error_covariance_matrix_of_predictions()
        kalman_gain_g = self.__compute_kalman_gain_for_state(sigma_g_y, sigma_y_y)
        self.__update_state_internal(kalman_gain_g)

    def __predict_at_obs(self, x_sample, log_params, g_mean):
        if self.learn_gp_parameters and self.learn_sigma:
            log_gp_params = log_params[:-1]
            log_sigma = log_params[-1]

        elif self.learn_gp_parameters:
            log_gp_params = log_params
            log_sigma = self.initial_log_sigma

        elif self.learn_sigma:
            log_gp_params = self.initial_log_gp_params
            log_sigma = log_params

        else:
            log_gp_params = self.initial_log_gp_params
            log_sigma = self.initial_log_sigma

        cov_func = SquaredExponential.from_parameters_vector(np.exp(log_gp_params))
        mean, _ = cov_func.predict(x_sample, self.inducing_points_locations, np.exp(log_sigma), g_mean)

        return mean

    def __sample_parameters(self):
        if self.params_dimensionality > 0:
            self.params_ensemble = self.params_ensemble + np.random.multivariate_normal(
                mean=np.zeros(shape=(self.params_dimensionality,)), cov=self.sigma_eta, size=self.ensemble_size)

    def __predict_observations(self, x_new):
        for ens_idx in range(self.ensemble_size):
            mean = self.__predict_at_obs(x_new, self.params_ensemble[ens_idx], self.g_ensemble[ens_idx])
            self.predictions[ens_idx] = mean

    def __compute_noisy_trajectories(self, f_new_noisy):
        for ens_idx in range(self.ensemble_size):
            self.y_trajectories[ens_idx] = (
                    f_new_noisy + np.random.normal(loc=0., scale=self.sigma_y, size=(self.sample_size,))).T

    def __compute_cross_covariance_of_parameters_and_predictions(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        params_ensemble_mean = np.mean(self.params_ensemble, axis=0)
        sigma_eta_y = np.zeros((self.params_dimensionality, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_eta_y += np.outer(self.params_ensemble[ens_idx] - params_ensemble_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_eta_y /= (self.ensemble_size - 1)

        return sigma_eta_y

    def __compute_forecast_error_covariance_matrix_of_predictions(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        sigma_y_y = np.zeros((self.sample_size, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_y_y += np.outer(self.predictions[ens_idx] - predictions_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_y_y /= (self.ensemble_size - 1)

        return sigma_y_y

    def __compute_kalman_gain_for_parameters(self, sigma_eta_y, sigma_y_y):
        kalman_gain_eta = np.matmul(sigma_eta_y, np.linalg.inv(sigma_y_y + self.sigma_y * np.eye(self.sample_size)))

        return kalman_gain_eta

    def __update_parameters_internal(self, kalman_gain_eta):
        for ens_idx in range(self.ensemble_size):
            self.params_ensemble[ens_idx] = self.params_ensemble[ens_idx] + np.matmul(kalman_gain_eta,
                                                                          self.y_trajectories[ens_idx] - self.predictions[ens_idx])
        self.__check_params_ensemble()

    def __compute_cross_covariance_of_state_and_prediction(self):
        predictions_mean = np.mean(self.predictions, axis=0)
        g_ensemble_mean = np.mean(self.g_ensemble, axis=0)
        sigma_g_y = np.zeros((self.grid_size, self.sample_size))
        for ens_idx in range(self.ensemble_size):
            sigma_g_y += np.outer(self.g_ensemble[ens_idx] - g_ensemble_mean, self.predictions[ens_idx] - predictions_mean)
        sigma_g_y /= (self.ensemble_size - 1)

        return sigma_g_y

    def __compute_kalman_gain_for_state(self, sigma_g_y, sigma_y_y):
        kalman_gain_g = np.matmul(sigma_g_y, np.linalg.inv(sigma_y_y + self.sigma_y * np.eye(self.sample_size)))

        return kalman_gain_g

    def __update_state_internal(self, kalman_gain_g):
        for ens_idx in range(self.ensemble_size):
            self.g_ensemble[ens_idx] = self.g_ensemble[ens_idx] + np.matmul(kalman_gain_g,
                                                                self.y_trajectories[ens_idx] - self.predictions[ens_idx])
        self.__check_g_ensemble()

    def __check_params_ensemble(self):
        if not np.all(np.isfinite(self.params_ensemble)):
            raise ValueError('params are not finite')
        self.params_ensemble = np.clip(self.params_ensemble, self.min_exp, None)

    def __check_g_ensemble(self):
        if not np.all(np.isfinite(self.g_ensemble)):
            raise ValueError('g are not finite')

    def get_eta_ensemble(self):
        """
        :return: Ensemble of logarithms of GP log hyperparameters
        """
        if self.learn_gp_parameters and self.learn_sigma:
            return self.params_ensemble[:, -1]

        elif self.learn_gp_parameters:
            return self.params_ensemble

    def get_log_mean_params(self):
        """
        :return: logarithm of mean GP hyperparameters, logarithm of mean noise variance
        """
        if self.learn_gp_parameters and self.learn_sigma:
            params = self.params_ensemble.mean(axis=0)
            log_gp_params = params[:-1]
            log_sigma = params[-1]

        elif self.learn_gp_parameters:
            log_gp_params = self.params_ensemble.mean(axis=0)
            log_sigma = self.initial_log_sigma

        elif self.learn_sigma:
            log_gp_params = self.initial_log_gp_params
            log_sigma = self.params_ensemble.mean(axis=0)

        else:
            log_gp_params = self.initial_log_gp_params
            log_sigma = self.initial_log_sigma

        return log_gp_params, log_sigma

    def get_g_mean(self):
        """
        :return: mean of state ensemble
        """
        return np.mean(self.g_ensemble, axis=0)

    def compute_nmse(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: NMSE between predicted and true values at sample points
        """
        eta_mean = np.mean(self.params_ensemble, axis=0)
        g_mean = np.mean(self.g_ensemble, axis=0)

        mean = self.__predict_at_obs(x_sample, eta_mean, g_mean)

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
