import networkx as nx
import numpy as np
import random
import scipy.stats as stats


def load_test_data(cluster_size=5, depth=1000, grid_size=101, single_sample=False):
    """ Simulate a toy Binomial data set.

    True tree: (4,(3,2(1,0)))
        4
    3        2
        1        0
    """
    random.seed(0)

    np.random.seed(0)

    def compute_log_likelihood(x, n, grid_size=grid_size):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, n, grid)

    graph = nx.DiGraph()
    graph.add_edge(-1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(2, 0)
    graph.add_edge(2, 1)

    cluster_params = [[0.1, 0.1, 0.9], [0.2, 0.1, 0.02], [0.3, 0.2, 0.92], [0.7, 0.8, 0.0], [1.0, 1.0, 1.0]]

    if single_sample:
        cluster_params = [[x[0], ] for x in cluster_params]

    data = []

    labels = []

    for i, params in enumerate(cluster_params):
        for _ in range(cluster_size):
            data_point = []

            n = stats.poisson.rvs(depth)

            for p in params:
                x = stats.binom.rvs(n, p)

                data_point.append(compute_log_likelihood(x, n))

            data.append(np.array(data_point))

            labels.append(i)

    return data, labels, graph
