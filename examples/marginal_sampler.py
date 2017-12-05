from __future__ import division

import networkx as nx
import numpy as np
import random
import scipy.stats as stats

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.kernels.marginal.data_structures import MarginalNode
from phyclone.kernels.marginal.bootstrap import MarginalBootstrapKernel
from phyclone.kernels.marginal.fully_adapted import MarginalFullyAdaptedKernel
from phyclone.kernels.marginal.utils import get_constrained_path, get_graph, get_nodes, sample_sigma, get_labels, get_tree
from phyclone.math_utils import discrete_rvs
from phyclone.tree import Tree
from phyclone.samplers.adaptive import AdaptiveSampler
from phyclone.samplers.particle_gibbs import ParticleGibbsSampler

from sklearn.metrics import homogeneity_completeness_v_measure


def main():
    random.seed(0)

    np.random.seed(0)

    data, labels, true_graph = simulate_binomial_data()

    alpha = 1e-3

    num_particles = 20

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    data_points = {0: range(len(data))}

    nodes = [MarginalNode(0, data[0].shape, []), ]

    tree = Tree(data_points, nodes)

#     kernel = MarginalBootstrapKernel(alpha, data[0].shape)

    kernel = MarginalFullyAdaptedKernel(alpha, data[0].shape)

    for i in range(1000):
        sigma = sample_sigma(tree)

        data_sigma = [data[data_idx] for data_idx in sigma]

        constrained_path = get_constrained_path(data, kernel, sigma, tree)

        sampler = ParticleGibbsSampler(constrained_path, data_sigma, kernel, num_particles, resample_threshold=0.5)

        swarm = sampler.sample()

        particle_idx = discrete_rvs(swarm.weights)

        particle = swarm.particles[particle_idx]

        tree = get_tree(particle, sigma)

#         kernel.alpha = conc_sampler.sample(alpha, len(tree.nodes), len(data))

        if i % 1 == 0:
            pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
            print
            print i, kernel.alpha
            print sigma
            print pred_labels
            print homogeneity_completeness_v_measure(labels, pred_labels), len(tree.nodes)
            print particle.log_w
            print nx.is_isomorphic(tree._graph, true_graph)
            print [x.idx for x in tree.roots]
            print

            for node_idx in tree.nodes:
                print node_idx, [(x + 1) / 100 for x in np.argmax(tree.nodes[node_idx].log_R, axis=1)]


def simulate_binomial_data():
    def compute_log_likelihood(x, n, grid_size=100):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, n, grid)

    eps = 1e-10
    graph = nx.DiGraph()
    graph.add_edge(-1, 5)
    graph.add_edge(5, 4)
    graph.add_edge(5, 3)
    graph.add_edge(3, 1)
    graph.add_edge(3, 2)
    clusters = [[0.1, 0.1, 0.9], [0.2, 0.1, 0.02], [0.3, 0.2, 0.92], [0.7, 0.8, 0.0], [1.0 - eps, 1.0 - eps, 1.0 - eps]]

    data = []

    labels = []

    for i, params in enumerate(clusters):
        for _ in range(5):
            data_point = []

            n = stats.poisson.rvs(10000)

            for p in params:
                x = stats.binom.rvs(n, p)

                data_point.append(compute_log_likelihood(x, n))

            data.append(np.array(data_point))

            labels.append(i)

    return data, labels, graph


if __name__ == '__main__':
    main()
