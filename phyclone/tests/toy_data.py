import numpy as np
import random
import scipy.stats as stats

from phyclone.data import DataPoint


def load_test_data(cluster_size, depth=1000, grid_size=101, single_sample=False):
    random.seed(0)

    np.random.seed(0)

    def compute_log_likelihood(x, n, grid_size=grid_size):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, n, grid)

    cluster_params = [[0.1, 0.1, 0.9], [0.7, 0.1, 0.02], [0.3, 0.8, 0.0], [1.0, 1.0, 1.0]]

    if single_sample:
        cluster_params = [[x[0], ] for x in cluster_params]

    data = []

    idx = 0

    for params in cluster_params:
        for _ in range(cluster_size):
            data_point = []

            for p in params:
                d = stats.poisson.rvs(depth)

                x = stats.binom.rvs(d, p)

                data_point.append(compute_log_likelihood(x, d))

            data.append(DataPoint(idx, np.array(data_point), outlier_prob=1e-4))

            idx += 1

    return data
