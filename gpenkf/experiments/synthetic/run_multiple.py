from gpenkf.experiments.experiment_runner import ExperimentRunner
from gpenkf.core.parameters import Parameters
from gpenkf.experiments.synthetic.data_generator import DataGenerator
from tqdm import trange
import numpy as np

if __name__ == "__main__":
    sample_size = 10
    data_provider = DataGenerator(borders=[-10, 10],
                                  sample_size=sample_size,
                                  f=lambda x: x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x),
                                  noise=0.01,
                                  validation_size=1000)

    grid_size = 51
    x = np.linspace(-10, 10, grid_size)
    x = np.expand_dims(x, axis=1)

    parameters = Parameters(T=200, sample_size=sample_size, grid_size=grid_size, inducing_points_locations=x, ensemble_size=100, sigma_eta=0.1,
                            sigma_y=0.1, init_cov=0.01, initial_log_gp_params=[0, 0], initial_log_sigma=0,
                            log_sigma_unlearnt=0, gp_hyperparams_dimensionality=2)

    num_rseeds = 10
    multi_runners = []
    algorithms = ['learn_enkf_gp', 'learn_enkf_liuwest_gp', 'learn_enkf_augmented_gp', 'learn_normal_gp']
    for rseed in range(num_rseeds):
        multi_runner = ExperimentRunner(data_provider=data_provider,
                              parameters=parameters,
                              algorithms=algorithms)
        multi_runners.append(multi_runner)

    for rseed in trange(num_rseeds):
        np.random.seed(rseed)
        multi_runners[rseed].run()

    for rseed in range(num_rseeds):
        multi_runners[rseed].save_results('results_multiple/{}'.format(rseed))