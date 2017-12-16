from __future__ import division, print_function

from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np
import random

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.mcmc.metropolis_hastings import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsSubtreeSampler
from phyclone.tree import get_single_node_tree

from toy_data import load_test_data
from phyclone.math_utils import log_normalize, discrete_rvs, exp_normalize


# TODO: Try doing a loop of outliers and attaching to tree nodes in a  Gibbs step
def resample_outliers(tree):
    outliers = list(tree.outliers)

    random.shuffle(outliers)

    for data_point in outliers:
        log_p = {-1: tree.log_p}

        tree.outliers.remove(data_point)

        for node in tree.nodes.values():
            node.add_data_point(data_point)

            tree._update_ancestor_nodes(node)

            log_p[node.idx] = tree.log_p

            node.remove_data_point(data_point)

            tree._update_ancestor_nodes(node)

        p, _ = exp_normalize(np.array(list(log_p.values())).astype(float))

        x = discrete_rvs(p)

        node_idx = list(log_p.keys())[x]

        if node_idx == -1:
            tree.outliers.append(data_point)

        else:
            tree.nodes[node_idx].add_data_point(data_point)

            tree._update_ancestor_nodes(tree.nodes[node_idx])

    return tree


data, labels, true_graph = load_test_data(cluster_size=5, outlier_size=2)

tree = get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

mh_sampler = PruneRegraphSampler()

pg_sampler = ParticleGibbsSubtreeSampler(
    data[0].shape,
    alpha=1.0,
    kernel='semi-adapted',
    num_particles=20,
    outlier_prob=1e-4,
    resample_threshold=0.5
)

print('Starting sampling')

num_iters = int(1e4)

trace = []

for i in range(num_iters):
    tree = pg_sampler.sample_tree(data, tree)

    tree = mh_sampler.sample_tree(data, tree)

    tree = resample_outliers(tree)

    pg_sampler.alpha = conc_sampler.sample(pg_sampler.alpha, len(
        tree.nodes), len([x for x in tree.labels.values() if x != -1]))

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        print()
        print(i, pg_sampler.alpha)
        print(pred_labels)
        print(homogeneity_completeness_v_measure(labels, pred_labels), len(tree.nodes))
        print(tree.log_p)
        print(nx.is_isomorphic(tree._graph, true_graph))
        print([x.idx for x in tree.roots])
        print()

        for node_idx in tree.nodes:
            print(node_idx, [(x + 1) / 101 for x in np.argmax(tree.nodes[node_idx].log_R, axis=1)])

        if i >= min((num_iters / 2), 1000):
            trace.append(tree)

consensus_tree = get_consensus_tree(trace)

print(consensus_tree.edges())

for node in consensus_tree.nodes():
    print(node, consensus_tree.nodes[node]['data_points'])
