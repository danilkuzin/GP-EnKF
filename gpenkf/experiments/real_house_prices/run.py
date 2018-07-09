from gpenkf.experiments.experiment_runner import ExperimentRunner
from gpenkf.core.parameters import Parameters
from gpenkf.experiments.real_house_prices.house_data import HouseData

import numpy as np

if __name__=="__main__":
    sample_size=1000
    data_provider = HouseData(sample_size=sample_size, validation_size=5000)
    data_provider.prepare()

    borders = np.array([[50, 55], [-6., 2.]])
    grid_size = 50

    x2 = np.linspace(borders[0, 0], borders[0, 1], grid_size)
    x1 = np.linspace(borders[1, 0], borders[1, 1], grid_size)
    x = np.stack(np.meshgrid(x1, x2), -1).reshape(-1, 2)
    grid_size =np.power(grid_size, x.shape[1])

    parameters = Parameters(T=50, sample_size=sample_size, grid_size=grid_size, inducing_points_locations=x, ensemble_size=500, sigma_eta=0.1,
                            sigma_y=.1, init_cov=1, initial_log_gp_params=[0, 0, 0], initial_log_sigma=0,
                            log_sigma_unlearnt=0, gp_hyperparams_dimensionality=3)

    runner = ExperimentRunner(data_provider=data_provider,
                              parameters=parameters,
                              algorithms=['learn_enkf_gp', 'learn_enkf_liuwest_gp', 'learn_enkf_augmented_gp'])
    runner.run()
    runner.save_results('results')