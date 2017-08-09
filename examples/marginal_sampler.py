from __future__ import division

import networkx as nx
import numpy as np
import random
import scipy.stats as stats

from fscrp.concentration import GammaPriorConcentrationSampler
from fscrp.kernels.marginal.fully_adapted import MarginalFullyAdaptedKernel
from fscrp.kernels.marginal.utils import get_constrained_path, get_graph, get_nodes, sample_sigma, get_labels
from fscrp.math_utils import discrete_rvs
from fscrp.samplers.adaptive import AdaptiveSampler
from fscrp.samplers.particle_gibbs import ParticleGibbsSampler

from sklearn.metrics import homogeneity_completeness_v_measure


def main():
    random.seed(0)

    np.random.seed(0)

    data, labels, true_graph = simulate_binomial_data()

    alpha = 1

    num_particles = 20

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    graph = nx.DiGraph()

    graph.add_edge(-1, 0)

    graph.node[-1]['data_points'] = []

    graph.node[0]['data_points'] = range(len(data))

    kernel = MarginalFullyAdaptedKernel(alpha, data[0].shape)

    for i in range(1000):
        sigma = sample_sigma(graph)

        data_sigma = [data[data_idx] for data_idx in sigma]

        constrained_path = get_constrained_path(data, graph, kernel, sigma)

        sampler = ParticleGibbsSampler(constrained_path, data_sigma, kernel, num_particles, resample_threshold=0.5)

        particles = sampler.sample()

        particle_idx = discrete_rvs([x[1] for x in particles])

        particle = particles[particle_idx][0]

        graph = get_graph(particle, sigma=sigma)

        kernel.alpha = conc_sampler.sample(alpha, len(set(get_labels(graph))), len(data))

        if i % 1 == 0:
            print
            print i, kernel.alpha
            print get_labels(graph)
            print homogeneity_completeness_v_measure(labels, get_labels(graph)), len(set(get_labels(graph)))
            print particle.log_w
            print graph.nodes()
            print graph.edges()
            print len(get_nodes(particle))
            print nx.is_isomorphic(graph, true_graph)
            print


def simulate_binomial_data():
    def compute_log_likelihood(x, n, grid_size=100):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, n, grid)

    graph = nx.DiGraph()
    graph.add_edge(-1, 5)
    graph.add_edge(5, 4)
    graph.add_edge(5, 3)
    graph.add_edge(3, 1)
    graph.add_edge(3, 2)
    clusters = [[0.1, 0.1, 0.9], [0.2, 0.1, 0.02], [0.3, 0.2, 0.92], [0.7, 0.8, 0.0], [1.0, 1.0, 1.0]]

    data = []

    labels = []

    for i, params in enumerate(clusters):
        for _ in range(2):
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
