import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle as pkl

from gpenkf.gp_util.squared_exponential import SquaredExponential
from gpenkf.experiments.real_house_prices.house_data import HouseData


def plot_res(res, params, dense_grid_size, f, borders, alg_name, f_std, f_mean):
    x2 = np.linspace(borders[0, 0], borders[0, 1], dense_grid_size)
    x1 = np.linspace(borders[1, 0], borders[1, 1], dense_grid_size)
    dense_grid = np.stack(np.meshgrid(x1, x2), -1).reshape(-1, 2)

    cov_func = SquaredExponential.from_parameters_vector(np.exp(res.eta_mean_history[-1]))
    mean, cov = cov_func.predict(dense_grid, params.x,
                                         np.exp(res.sigma_mean_history[-1]), res.g_mean_history[-1])

    mean = np.exp(mean * f_std + f_mean)

    fig, ax = plt.subplots(figsize=(10, 20))
    m = Basemap(resolution='h',  # c, l, i, h, f or None
                projection='merc', llcrnrlon=borders[1, 0], llcrnrlat=borders[0, 0],
                urcrnrlon=borders[1, 1], urcrnrlat=borders[0, 1], ax=ax)

    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    m.drawcoastlines()

    xc1, xc2 = m(dense_grid[:, 0], dense_grid[:, 1])
    xc1 = xc1.reshape((dense_grid_size, dense_grid_size))
    xc2 = xc2.reshape((dense_grid_size, dense_grid_size))
    mean = mean.reshape((dense_grid_size, dense_grid_size))
    cs = m.contour(xc1, xc2, mean, linewidths=1.5, ax=ax)

    cbar = m.colorbar(cs)
    cbar.set_label('price')

    plt.savefig('graphics/{}_mean.pdf'.format(alg_name), format='pdf')

    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 20))
    m = Basemap(resolution='c',
                projection='merc', llcrnrlon=borders[1, 0], llcrnrlat=borders[0, 0],
                urcrnrlon=borders[1, 1], urcrnrlat=borders[0, 1], ax=ax)

    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    m.drawcoastlines()

    cov = np.diag(cov)
    cov = cov.reshape((dense_grid_size, dense_grid_size))
    cs = m.contour(xc1, xc2, cov, linewidths=1.5, ax=ax)

    cbar = m.colorbar(cs)
    cbar.set_label('cov')

    plt.savefig('graphics/{}_var.pdf'.format(alg_name), format='pdf')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.plot(res.nmse_history)
    plt.savefig('graphics/{}_nmse.pdf'.format(alg_name), format='pdf')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.plot(res.likelihood_history)
    plt.savefig('graphics/{}_likelihood.pdf'.format(alg_name), format='pdf')


if __name__=="__main__":
    with open('results/results.pkl', "rb") as f:
        results = pkl.load(f)
    with open('results/parameters.pkl', "rb") as f:
        parameters = pkl.load(f)

    data_provider = HouseData(sample_size=100, validation_size=50)
    data_provider.prepare()

    for result_key in results.keys():
        plot_res(res=results[result_key], params=parameters, dense_grid_size=100,
                 f=lambda x: x / 2 + (25 * x) / (1 + x ** 2) * np.cos(x), borders=np.array([[50, 55], [-6., 2.]]),
                 alg_name=result_key, f_std=data_provider.prices_std, f_mean=data_provider.prices_mean)