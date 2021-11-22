""" Simple example to show the single pass SMC algorithm.

This example runs two passes of the SMC. The first has a good ordering which yields the correct tree. The second has a
bad ordering which yields the incorrect tree.
"""

from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.math_utils import discrete_rvs
from phyclone.smc.kernels import FullyAdaptedKernel
from phyclone.smc.samplers import SMCSampler
from phyclone.tree import FSCRPDistribution, TreeJointDistribution

from toy_data import load_test_data


def main():
    data_points, true_tree = load_test_data(cluster_size=2, depth=int(1e5), outlier_size=1, shuffle=False)

    print("Good ordering")
    sampler, pred_tree = sample(data_points)
    print_stats(pred_tree, true_tree, sampler)

    print()
    print("#" * 100)
    print()

    print("Bad ordering")
    sampler, pred_tree = sample(data_points[::-1])
    print_stats(pred_tree, true_tree, sampler)


def sample(data_points):
    tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

    kernel = FullyAdaptedKernel(tree_dist, outlier_proposal_prob=0.1)
    
    smc_sampler = SMCSampler(
        data_points,
        kernel,
        num_particles=100,
        resample_threshold=0.5
    )
    
    swarm = smc_sampler.sample()

    idx = discrete_rvs(swarm.weights)

    particle = swarm.particles[idx]

    tree = particle.tree

    tree.relabel_nodes()

    return smc_sampler, tree


def print_stats(pred_tree, true_tree, sampler):
    pred_labels = [pred_tree.labels[x] for x in sorted(pred_tree.labels)]
    true_labels = [true_tree.labels[x] for x in sorted(true_tree.labels)]

    print(pred_labels)
    print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(pred_tree.nodes))
    print(sampler.kernel.tree_dist.log_p_one(pred_tree))
    print(nx.is_isomorphic(pred_tree.graph, true_tree.graph))
    print([x for x in pred_tree.roots])
    print(pred_tree.graph.edges)
    for node_idx in pred_tree.nodes:
        print(node_idx, [(x + 1) / 101 for x in np.argmax(pred_tree.graph.nodes[node_idx]["log_R"], axis=1)])


if __name__ == "__main__":
    main()
