import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gpenkf.gp_util.squared_exponential import SquaredExponential


def plot_res(res, params, dense_grid_size, f, borders, alg_name):
    dense_grid = np.linspace(borders[0], borders[1], dense_grid_size)

    cov_func = SquaredExponential.from_parameters_vector(np.exp(res.eta_mean_history[-1]))
    mean, cov = cov_func.predict(np.expand_dims(dense_grid, axis=1), params.x,
                                         np.exp(res.sigma_mean_history[-1]), res.g_mean_history[-1])

    lower = mean - 2 * np.sqrt(np.diag(cov))
    upper = mean + 2 * np.sqrt(np.diag(cov))

    df = pd.DataFrame({'iter': range(1, params.T + 1)})
    fig, ax = plt.subplots(1)
    ax.fill_between(dense_grid, lower, upper, color='aquamarine', edgecolor='blue')
    ax.plot(dense_grid, mean, 'b', label='mean')
    ax.plot(dense_grid, f(dense_grid), 'r', label='true f')
    ax.set_xlabel('x')
    ax.legend()
    plt.savefig('graphics_new_single/{}_predictions.eps'.format(alg_name), format='eps')

    if not np.any(np.isnan(res.eta_last_ensemble)):
        fig, ax = plt.subplots(1)
        sns.distplot(res.eta_last_ensemble[:, 0], label='log variance', ax=ax)
        sns.distplot(res.eta_last_ensemble[:, 1], label='log lengthscale', ax=ax)
        ax.legend()
        plt.savefig('graphics_new_single/{}_eta_distplot.eps'.format(alg_name), format='eps')


if __name__=="__main__":
    with open('results/results.pkl', "rb") as f:
        results = pkl.load(f)
    with open('results/parameters.pkl', "rb") as f:
        parameters = pkl.load(f)

    for result_key in results.keys():
            plot_res(res=results[result_key], params=parameters, dense_grid_size=250,
                     f=lambda x: x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x), borders=[-10, 10], alg_name=result_key)
