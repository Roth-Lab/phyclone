import networkx as nx
import numpy as np
# import random
import scipy.stats as stats

from phyclone.data.base import DataPoint
from phyclone.tree import Tree
from phyclone.data.pyclone import compute_outlier_prob


def load_test_data(rng, cluster_size=5, depth=1000, grid_size=101, outlier_size=2, shuffle=True, single_sample=False):
    """ Simulate a toy Binomial data set.

    True tree: (4,(3,2(1,0)))
        4
    3        2
        1        0
    """
    # random.seed(0)
    #
    # np.random.seed(0)

    def compute_log_likelihood(x, d, grid_size=grid_size):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, d, grid)

    graph = nx.DiGraph()
    graph.add_edge(-1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(2, 0)
    graph.add_edge(2, 1)

    cluster_params = [
        [0.1, 0.2, 0.9, 0.05],
        [0.2, 0.1, 0.02, 0.9],
        [0.3, 0.3, 0.92, 0.95],
        [0.69, 0.69, 0.07, 0.04],
        [0.99, 0.99, 0.99, 0.99],
        [0.01, 0.95, 0.95, 0.01]
    ]

    if single_sample:
        cluster_params = [[x[0], ] for x in cluster_params]

#     else:
#         cluster_params = [[x[0], x[1], x[2]] for x in cluster_params]

    data = []

    idx = 0

    tree = Tree((len(cluster_params[0]), grid_size), )

    node_0 = tree.create_root_node([], [])
    node_1 = tree.create_root_node([], [])
    node_2 = tree.create_root_node([node_0, node_1], [])
    node_3 = tree.create_root_node([], [])
    node_4 = tree.create_root_node([node_2, node_3], [])

    for node, params in enumerate(cluster_params):
        if node == 5:
            n = outlier_size

            outlier_prob = 1e-2

        else:
            n = cluster_size

            outlier_prob = 1e-4

        for _ in range(n):
            data_point = []

            for p in params:
                d = stats.poisson.rvs(depth, random_state=rng)

                x = stats.binom.rvs(d, p, random_state=rng)

                data_point.append(compute_log_likelihood(x, d))

            out_probs = compute_outlier_prob(outlier_prob, n)
            data.append(DataPoint(idx, np.array(data_point), outlier_prob=out_probs[0],
                                  outlier_prob_not=out_probs[1]))

            if node == 5:
                tree.add_data_point_to_outliers(data[-1])

            else:
                tree.add_data_point_to_node(data[-1], node)

            idx += 1
    
    if shuffle:
        rng.shuffle(data)

    return data, tree
