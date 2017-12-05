from __future__ import division, print_function

from math import log

import matplotlib.pyplot as pp
import networkx as nx
import numpy as np
import random
import scipy.stats as stats

from sklearn.metrics import homogeneity_completeness_v_measure


from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.kernels.marginal.data_structures import MarginalNode
from phyclone.kernels.marginal.bootstrap import MarginalBootstrapKernel
from phyclone.kernels.marginal.fully_adapted import MarginalFullyAdaptedKernel
from phyclone.kernels.marginal.utils import get_constrained_path, get_graph, get_nodes, sample_sigma, get_labels, get_tree
from phyclone.math_utils import discrete_rvs
from phyclone.samplers.adaptive import AdaptiveSampler
from phyclone.samplers.particle_gibbs import ParticleGibbsSampler
from phyclone.samplers.swarm import ParticleSwarm
from phyclone.tree import Tree


def load_tree(num_iters):
    random.seed(0)

    np.random.seed(0)

    data, labels, true_graph = simulate_binomial_data()

    data_points = {0: range(len(data))}

    nodes = [MarginalNode(0, data[0].shape, []), ]

    tree = Tree(data_points, nodes)

    return data, labels, true_graph, tree


def simulate_binomial_data():
    def compute_log_likelihood(x, n, grid_size=101):
        eps = 1e-10

        grid = np.linspace(0 + eps, 1 - eps, grid_size)

        return stats.binom.logpmf(x, n, grid)

    graph = nx.DiGraph()
    graph.add_edge(-1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(2, 0)
    graph.add_edge(2, 1)
    clusters = [[0.1, 0.1, 0.9], [0.2, 0.1, 0.02], [0.3, 0.2, 0.92], [0.7, 0.8, 0.0], [1.0, 1.0, 1.0]]

    data = []

    labels = []

    for i, params in enumerate(clusters):
        for _ in range(10):
            data_point = []

            n = stats.poisson.rvs(100000)

            for p in params:
                x = stats.binom.rvs(n, p)

                data_point.append(compute_log_likelihood(x, n))

            data.append(np.array(data_point))

            labels.append(i)

    return data, labels, graph


def propose_prune_regraph(old_tree):
    if len(old_tree.nodes) == 1:
        return old_tree

    new_tree = old_tree.copy()

    nodes = new_tree.nodes.values()

    subtree_root = random.choice(nodes)

    subtree = new_tree.get_subtree(subtree_root)

    new_tree.remove_subtree(subtree)

    remaining_nodes = new_tree.nodes.values()

    if len(remaining_nodes) == 0:
        return old_tree

    parent = random.choice(remaining_nodes)

    new_tree.add_subtree(subtree, parent)

    old_log_p = old_tree.log_p_one

    new_log_p = new_tree.log_p_one

    u = random.random()

    if new_log_p - old_log_p > log(u):
        print('Accepting prune regraph')

        tree = new_tree

    else:
        tree = old_tree

    return tree


def run_particle_gibbs_sampler(data, kernel, tree, num_particles=10, resample_threshold=0.5):
    sigma = sample_sigma(tree)

    data_sigma = [data[data_idx] for data_idx in sigma]

    constrained_path = get_constrained_path(data, kernel, sigma, tree)

    sampler = ParticleGibbsSampler(
        constrained_path,
        data_sigma,
        kernel,
        num_particles=num_particles,
        resample_threshold=resample_threshold
    )

    return sampler.sample(), sigma


def run_subtree_particle_gibbs_sampler(data, kernel, tree, num_particles=10, resample_threshold=0.5):

    node_idxs = tree.nodes.keys()

    subtree_root_idx = random.choice(node_idxs)

    parent_node = tree.get_parent_node(tree.nodes[subtree_root_idx])

    if parent_node.idx == -1:
        parent_node = None

    subtree = tree.get_subtree(tree.nodes[subtree_root_idx])

    tree.remove_subtree(subtree)

    subtree.relabel_nodes(0)

    swarm, sigma = run_particle_gibbs_sampler(data, kernel, subtree, num_particles=num_particles)

    trees = []

    new_swarm = ParticleSwarm()

    for p, w in zip(swarm.particles, swarm.unnormalized_log_weights):
        if len(tree.nodes) == 0:
            min_node_idx = 0

        else:
            min_node_idx = max(tree.nodes.keys()) + 1

        w -= p.state.log_p_one

        t = get_tree(p, sigma=sigma)

        t.relabel_nodes(min_value=min_node_idx)

        tree.add_subtree(t, parent=parent_node)

        w += tree.log_p_one

        new_swarm.add_particle(w, t)

        tree.remove_subtree(t)

    particle_idx = discrete_rvs(new_swarm.weights)

    p = swarm.particles[particle_idx]

    t = get_tree(p, sigma=sigma)

    t.relabel_nodes(min_node_idx)

    tree.add_subtree(t, parent=parent_node)

    tree.relabel_nodes(0)

    return tree, sigma

# %%
data, labels, true_graph, tree = load_tree(10)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

alpha = 1

# %%
for i in range(1000):
    if i % 10 == 0:
        kernel = MarginalBootstrapKernel(alpha, data[0].shape)

        swarm, sigma = run_particle_gibbs_sampler(data, kernel, tree, num_particles=20)

        particle_idx = discrete_rvs(swarm.weights)

        p = swarm.particles[particle_idx]

        tree = get_tree(p, sigma=sigma)

    else:
        kernel = MarginalFullyAdaptedKernel(alpha, data[0].shape)

        tree, sigma = run_subtree_particle_gibbs_sampler(data, kernel, tree, num_particles=20)

    tree = propose_prune_regraph(tree)

    alpha = conc_sampler.sample(alpha, len(tree.nodes), len(data))

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        print()
        print(i, kernel.alpha)
        print(sigma)
        print(pred_labels)
        print(homogeneity_completeness_v_measure(labels, pred_labels), len(tree.nodes))
        print(tree.log_p)
        print(nx.is_isomorphic(tree._graph, true_graph))
        print([x.idx for x in tree.roots])
        print()

        for node_idx in tree.nodes:
            print(node_idx, [(x + 1) / 101 for x in np.argmax(tree.nodes[node_idx].log_R, axis=1)])
