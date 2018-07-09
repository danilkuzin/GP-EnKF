from gpenkf.experiments.experiment_runner import ExperimentRunner
from gpenkf.core.parameters import Parameters
from gpenkf.experiments.synthetic.data_generator import DataGenerator

import numpy as np

if __name__ == "__main__":
    sample_size = 5
    data_provider = DataGenerator(borders=[-10, 10],
                                  sample_size=sample_size,
                                  f=lambda x: x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x),
                                  noise=0.01,
                                  validation_size=10)

    grid_size = 51
    x = np.linspace(-10, 10, grid_size)
    x = np.expand_dims(x, axis=1)

    parameters = Parameters(T=200, sample_size=sample_size, grid_size=grid_size, inducing_points_locations=x, ensemble_size=100, sigma_eta=0.1,
                            sigma_y=0.1, init_cov=0.01, initial_log_gp_params=[0, 0], initial_log_sigma=0,
                            log_sigma_unlearnt=0, gp_hyperparams_dimensionality=2)

    runner = ExperimentRunner(data_provider=data_provider,
                              parameters=parameters,
                              algorithms=['learn_enkf_gp', 'learn_enkf_liuwest_gp', 'learn_enkf_augmented_gp', 'learn_normal_gp'])
    runner.run()
    runner.save_results('results')