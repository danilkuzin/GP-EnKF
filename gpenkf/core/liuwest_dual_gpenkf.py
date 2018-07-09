"""
Liu-West Dual GP-EnKF model
"""

import numpy as np
from gpenkf.core import DualGPEnKF


class LiuWestDualGPEnKF(DualGPEnKF):
    """
    LiuWest Dual GP-EnKF model. Has separate ensembles for parameters and state.
    Parameter evolution is modelled with Liu-West filter.

    :param parameters: The :class:`~gpenkf.core.parameters.parameters` parameters
    :param learn_gp_parameters: indicator if the GP hyperparameters should be learnt
    :param learn_sigma: indicator if the noise variance should be learnt
    """
    def __init__(self, parameters, learn_gp_parameters=True, learn_sigma=False, liu_west_delta=0.99):

        super().__init__(parameters, learn_gp_parameters, learn_sigma)

        if liu_west_delta > 1 or liu_west_delta <=0:
            raise Exception("liu_west_delta should be in (0, 1]")
        self.liu_west_delta = liu_west_delta

    def __sample_parameters(self):
        if self.params_dimensionality > 0:
            params_ensemble_mean = np.mean(self.params_ensemble, axis=0)
            params_ensemble_var = np.var(self.params_ensemble, axis=0)

            liu_west_a = (3. * self.liu_west_delta - 1.) / (2. * self.liu_west_delta)
            liu_west_h2 = 1.-liu_west_a**2

            params_ensemble_new_sample_mean = liu_west_a * self.params_ensemble \
                                              + (1-liu_west_a) * params_ensemble_mean

            params_ensemble_new_sample_variance = liu_west_h2 * params_ensemble_var

            for e in range(self.ensemble_size):
                self.params_ensemble[e] = np.random.multivariate_normal(
                    mean=params_ensemble_new_sample_mean[e], cov=np.diag(np.sqrt(params_ensemble_new_sample_variance)))


