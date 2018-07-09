"""
Classic GP
"""
import GPy
import numpy as np
from gpenkf.gp_util.squared_exponential import SquaredExponential


class NormalGP:
    """
    Classic GP model. Wrapper of GPy functions without inducing points

    :param parameters: The :class:`~gpenkf.core.parameters.parameters` parameters
    """
    def __init__(self, parameters):

        self.inducing_points_locations = parameters.inducing_points_locations
        self.initial_log_gp_params = parameters.initial_log_gp_params
        self.initial_log_sigma = parameters.initial_log_sigma

        self.x_history = np.empty((0, self.inducing_points_locations.shape[1]))
        self.f_history = np.empty((0,))

        self.m = None
        self.g = None
        self.params = np.zeros((3,))

    def compute_nmse(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: NMSE between predicted and true values at sample points
        """
        mean, _ = np.squeeze(self.m.predict(x_sample))

        return np.mean(np.sqrt((mean-f_true_sample)**2) / np.sqrt(f_true_sample**2))

    def compute_log_likelihood(self, x_sample, f_true_sample):
        """
        :param x_sample: location of the points to predict
        :param f_true_sample: true value at sample points
        :return: log likelihood of true values at sample points given the estimated model
        """
        cov_func = SquaredExponential(variance=np.exp(self.params[0]), lengthscale=np.exp(self.params[1]))
        return cov_func.log_likelihood(f_true_sample, x_sample, np.exp(self.params[2]))

    def run_iteration(self, x_new, f_new_noisy):
        """
        Perform one iteration learning parameters with sample data.

        :param x_new: locations of new observations
        :param f_new_noisy: values of new observations
        """
        self.x_history = np.append(self.x_history, x_new, axis=0)
        self.f_history = np.append(self.f_history, f_new_noisy, axis=0)
        kernel = GPy.kern.RBF(input_dim=self.inducing_points_locations.shape[1], variance=np.exp(self.initial_log_gp_params[0]),
                              lengthscale=np.exp(self.initial_log_gp_params[1]))

        self.m = GPy.models.GPRegression(self.x_history, np.expand_dims(self.f_history, axis=1), kernel)
        self.m.Gaussian_noise.variance = np.exp(self.initial_log_sigma)

        self.m.constrain_positive()

        self.m.optimize()
        self.m.Gaussian_noise.variance = np.exp(self.initial_log_sigma)
        self.params[0] = np.log(self.m.kern.param_array[0])
        self.params[1] = np.log(self.m.kern.param_array[1])
        self.params[2] = np.log(self.m.likelihood.variance[0])

        self.g, _ = np.squeeze(self.m.predict(self.inducing_points_locations))

    def get_g_mean(self):
        """
        :return: mean of state ensemble
        """
        return self.g

    def get_eta_ensemble(self):
        """
        For compatibility with other models.

        :return: nans
        """
        return np.nan

    def get_log_mean_params(self):
        """
        :return: logarithm of mean GP hyperparameters, logarithm of mean noise variance
        """
        return self.params[:-1], self.params[-1]
