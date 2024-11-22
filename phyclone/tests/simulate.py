import numpy as np
import scipy.stats as stats

from phyclone.data import compute_outlier_prob
from phyclone.data.base import DataPoint


def simulate_binomial_data(idx, n, p, rng, outlier_prob=0.0):
    p = np.atleast_1d(p)

    data = []

    for p_i in p:
        x = stats.binom.rvs(n, p_i, random_state=rng)

        data.append(log_binomial_likelihood(n, x))

    data = np.atleast_2d(data)

    dp_outlier_prob, dp_outlier_prob_not = compute_outlier_prob(outlier_prob, 1)

    return DataPoint(idx, data, outlier_prob=dp_outlier_prob, outlier_prob_not=dp_outlier_prob_not)


def log_binomial_likelihood(n, x, eps=1e-10, grid_size=101):
    grid = np.linspace(0 + eps, 1 - eps, grid_size)

    return stats.binom.logpmf(x, n, grid)
