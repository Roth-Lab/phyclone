from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.mcmc.gibbs_mh import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.smc.kernels import FullyAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution

from toy_data import load_test_data
from phyclone.math_utils import simple_log_factorial
from math import inf
from phyclone.run import instantiate_and_seed_RNG

rng = instantiate_and_seed_RNG(1234)

data, true_tree = load_test_data(rng, cluster_size=2, single_sample=True)

factorial_arr = np.full(len(data)+1, -inf)
simple_log_factorial(len(data), factorial_arr)

tree = Tree.get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)

tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

mh_sampler = PruneRegraphSampler(tree_dist, rng=rng)

kernel = FullyAdaptedKernel(tree_dist, outlier_proposal_prob=0.1, rng=rng)

pg_sampler = ParticleGibbsTreeSampler(
    kernel,
    num_particles=10,
    resample_threshold=0.5,
    rng=rng
)

num_iters = 100

trace = []

for i in range(num_iters):
    tree = pg_sampler.sample_tree(tree)

    tree = mh_sampler.sample_tree(tree)

    node_sizes = []

    for node, node_data in tree.node_data.items():
        if node == -1:
            continue

        node_sizes.append(len(node_data))

    tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(tree.nodes), sum(node_sizes))

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        true_labels = [true_tree.labels[x] for x in sorted(tree.labels)]
        print()
        print(i, tree_dist.prior.alpha)
        print(pred_labels)
        print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
        print(tree_dist.log_p_one(tree))
        print(nx.is_isomorphic(tree.graph, true_tree.graph))
        print([x for x in tree.roots])
        print()

        for node_idx in tree.nodes:
            print(node_idx, [(x + 1) / 101 for x in np.argmax(tree._graph.nodes[node_idx]["log_R"], axis=1)])

    if i >= num_iters / 2:
        trace.append(tree)

consensus_tree = get_consensus_tree(trace, data=data)

print(consensus_tree.edges())

for node in consensus_tree.nodes():
    print(node, consensus_tree.nodes[node]['idxs'])
