from __future__ import division, print_function

from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.metropolis_hastings import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.tree import get_single_node_tree

from toy_data import load_test_data
from phyclone.smc.utils import Configuration

data, labels, true_graph = load_test_data(cluster_size=5)

tree = get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

mh_sampler = PruneRegraphSampler()

pg_sampler = ParticleGibbsTreeSampler(
    data[0].shape,
    alpha=1.0,
    kernel='semi-adapted',
    num_particles=10,
    resample_threshold=0.5
)

tree = Configuration([], tree)

for i in range(1000):
    tree = pg_sampler.sample_tree(data, tree)

    tree = mh_sampler.sample_tree(data, tree)

#     pg_sampler.alpha = conc_sampler.sample(pg_sampler.alpha, len(tree.nodes), len(data))

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        print(pred_labels)
        print()
        print(i, pg_sampler.alpha)
        print(pred_labels)
        print(homogeneity_completeness_v_measure(labels, pred_labels), len(tree.tree.nodes))
        print(tree.log_p)
#         print(nx.is_isomorphic(tree._graph, true_graph))
        print([x.idx for x in tree.tree.roots])
        print()

        for node_idx in tree.tree.nodes:
            print(node_idx, [(x + 1) / 101 for x in np.argmax(tree.tree.nodes[node_idx].log_R, axis=1)])
