import os
import time

import numpy as np
from tqdm import tqdm

from gpenkf.core import DualGPEnKF, LiuWestDualGPEnKF, AugmentedGPEnKF, NormalGP

import pickle as pkl


class Results(object):
    def __init__(self, T, params_dimensionality, grid_size, ensemble_size):
        self.eta_mean_history = np.zeros((T, params_dimensionality))
        self.sigma_mean_history = np.zeros((T))
        self.g_mean_history = np.zeros((T, grid_size))
        self.likelihood_history = np.zeros((T,))
        self.nmse_history = np.zeros((T,))
        self.time = np.zeros((T,))
        self.eta_last_ensemble = np.zeros((ensemble_size, params_dimensionality))


class ExperimentRunner(object):
    def __init__(self, data_provider, parameters, algorithms):
        self.parameters = parameters
        self.algorithms = algorithms

        self.data_provider = data_provider

        self.runners = {}
        self.results = {}
        if 'learn_enkf_all' in self.algorithms:
            self.runners['gpenkf_learn_all'] = DualGPEnKF(parameters=self.parameters,
                                                          learn_gp_parameters=True,
                                                          learn_sigma=True)
            self.results['gpenkf_learn_all'] = Results(T=self.parameters.T,
                                                       params_dimensionality=self.parameters.hyperparams_dimensionality + 1,
                                                       grid_size=self.parameters.grid_size,
                                                       ensemble_size=self.parameters.ensemble_size)

        if 'learn_enkf_gp' in self.algorithms:
            self.runners['gpenkf_learn_gp'] = DualGPEnKF(parameters=self.parameters,
                                                         learn_gp_parameters=True,
                                                         learn_sigma=False)

            self.results['gpenkf_learn_gp'] = Results(T=self.parameters.T,
                                                      params_dimensionality=self.parameters.hyperparams_dimensionality,
                                                      grid_size=self.parameters.grid_size,
                                                      ensemble_size=self.parameters.ensemble_size)

        if 'learn_enkf_liuwest_gp' in self.algorithms:
            self.runners['gpenkf_learn_liuwest_gp'] = LiuWestDualGPEnKF(parameters=self.parameters,
                                                                        learn_gp_parameters=True,
                                                                        learn_sigma=False,
                                                                        liu_west_delta=0.95)

            self.results['gpenkf_learn_liuwest_gp'] = Results(T=self.parameters.T,
                                                              params_dimensionality=self.parameters.hyperparams_dimensionality,
                                                              grid_size=self.parameters.grid_size,
                                                              ensemble_size=self.parameters.ensemble_size)

        if 'learn_enkf_augmented_gp' in self.algorithms:
            self.runners['gpenkf_augmented_gp'] = AugmentedGPEnKF(parameters=self.parameters,
                                                                  learn_gp_parameters=True,
                                                                  learn_sigma=False)

            self.results['gpenkf_augmented_gp'] = Results(T=self.parameters.T,
                                                          params_dimensionality=self.parameters.hyperparams_dimensionality,
                                                          grid_size=self.parameters.grid_size,
                                                          ensemble_size=self.parameters.ensemble_size)

        if 'learn_normal_gp' in self.algorithms:
            self.runners['normal_gp'] = NormalGP(parameters=self.parameters)

            self.results['normal_gp'] = Results(T=self.parameters.T,
                                                params_dimensionality=self.parameters.hyperparams_dimensionality,
                                                grid_size=self.parameters.grid_size,
                                                ensemble_size=1)

    def run(self):
        for t in tqdm(range(self.parameters.T)):
            x_new, f_new_noisy = self.data_provider.generate_sample()

            for runner_key in self.runners.keys():
                start_time = time.time()
                self.runners[runner_key].run_iteration(x_new, f_new_noisy)
                self.results[runner_key].time[t] = time.time() - start_time
                self.results[runner_key].eta_mean_history[t], self.results[runner_key].sigma_mean_history[t] = \
                    self.runners[runner_key].get_log_mean_params()
                self.results[runner_key].g_mean_history[t] = self.runners[runner_key].get_g_mean().T

                self.results[runner_key].likelihood_history[t] = self.runners[runner_key].compute_log_likelihood(
                    self.data_provider.x_validation, self.data_provider.f_validation)
                self.results[runner_key].nmse_history[t] = self.runners[runner_key].compute_nmse(
                    self.data_provider.x_validation, self.data_provider.f_validation)

        for runner_key in self.runners.keys():
            self.results[runner_key].eta_last_ensemble = self.runners[runner_key].get_eta_ensemble()

    def save_results(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        with open('{}/results.pkl'.format(path), "wb") as f:
            pkl.dump(self.results, f)
        with open('{}/parameters.pkl'.format(path), "wb") as f:
            pkl.dump(self.parameters, f)
