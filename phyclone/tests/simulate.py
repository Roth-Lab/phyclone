import numpy as np
import scipy.stats as stats

from phyclone.data import DataPoint


def simulate_binomial_data(idx, n, p, outlier_prob=0.0):
    p = np.atleast_1d(p)

    data = []

    for p_i in p:
        x = stats.binom.rvs(n, p_i)

        data.append(log_binomial_likelihood(n, x))

    data = np.atleast_2d(data)

    return DataPoint(idx, data, outlier_prob=outlier_prob)


def log_binomial_likelihood(n, x, eps=1e-10, grid_size=101):
    grid = np.linspace(0 + eps, 1 - eps, grid_size)

    return stats.binom.logpmf(x, n, grid)
