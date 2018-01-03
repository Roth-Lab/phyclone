from __future__ import division, print_function

from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.mcmc import ParticleGibbsSubtreeSampler, OutlierSampler, PruneRegraphSampler
from phyclone.tree import Tree

from toy_data import load_test_data


data, labels, true_graph = load_test_data(cluster_size=1, depth=int(1e5), outlier_size=0, single_sample=False)

tree = Tree.get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

mh_sampler = PruneRegraphSampler()

outlier_sampler = OutlierSampler()

pg_sampler = ParticleGibbsSubtreeSampler(kernel='semi-adapted', num_particles=20, resample_threshold=0.5)

print('Starting sampling')

num_iters = int(1e4)

trace = []

for i in range(num_iters):
    tree = pg_sampler.sample_tree(data, tree)

    tree = mh_sampler.sample_tree(data, tree)

    tree = outlier_sampler.sample_tree(tree)

    tree.alpha = conc_sampler.sample(tree.alpha, tree.num_nodes, sum(tree.node_sizes.values()))

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        print()
        print(i, tree.alpha)
        print(pred_labels)
        print(homogeneity_completeness_v_measure(labels, pred_labels), len(tree.nodes))
        print(tree.log_p_one)
        print(nx.is_isomorphic(tree._graph, true_graph))
        print([x.idx for x in tree.roots])
        print(tree.graph.edges)
        print()

        for node_idx in tree.nodes:
            print(node_idx, [(x + 1) / 101 for x in np.argmax(tree.nodes[node_idx].log_R, axis=1)])

        if i >= min((num_iters / 2), 1000):
            trace.append(tree)

consensus_tree = get_consensus_tree(trace)

print(consensus_tree.edges())

for node in consensus_tree.nodes():
    print(node, consensus_tree.nodes[node]['data_points'])
