"""
Parameters set for all GP models.
"""
import numpy as np


class Parameters(object):
    """
        :param T: number of sample data sets
        :param sample_size: number of data points in every sample data set
        :param grid_size: size of inducing points grid
        :param inducing_points_locations:
        :param ensemble_size:
        :param sigma_eta: variance for parameter evolution
        :param sigma_y: variance for noise in trajectories
        :param init_cov: variance for generating initial ensemble
        :param initial_log_gp_params:
        :param initial_log_sigma:
        :param log_sigma_unlearnt:
        :param gp_hyperparams_dimensionality:
    """
    def __init__(self, T, sample_size, grid_size, inducing_points_locations, ensemble_size, sigma_eta,
                            sigma_y, init_cov, initial_log_gp_params, initial_log_sigma,
                            log_sigma_unlearnt, gp_hyperparams_dimensionality):
        self.sample_size = sample_size
        self.T = T
        self.grid_size = grid_size
        self.inducing_points_locations = inducing_points_locations
        self.hyperparams_dimensionality = gp_hyperparams_dimensionality
        self.ensemble_size = ensemble_size

        self.sigma_eta_learn_all = 0.1 * np.eye(3, dtype=np.double)
        self.sigma_eta_learn_gp = 0.1 * np.eye(2, dtype=np.double)
        self.sigma_eta = sigma_eta
        self.sigma_y = sigma_y
        self.init_cov = init_cov
        self.initial_log_gp_params = initial_log_gp_params
        self.initial_log_sigma = initial_log_sigma
        self.log_sigma_unlearnt = log_sigma_unlearnt
