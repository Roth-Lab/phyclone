from __future__ import division, print_function

from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np
import random

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.math_utils import discrete_rvs
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler, ParticleGibbsSubtreeSampler
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.kernels import SemiAdaptedKernel
from phyclone.tree import Tree, FSCRPDistribution
from phyclone.smc.utils import RootPermutationDistribution

import phyclone.mcmc.metropolis_hastings as mh

from toy_data import load_test_data


def main():
    data, labels, true_graph = load_test_data(cluster_size=2, depth=int(1e6), outlier_size=0, single_sample=False)

    tree = init_tree(data)

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    mh_sampler = mh.PruneRegraphSampler()

    mh_sampler2 = mh.NodeSwap()

    outlier_node_sampler = mh.OutlierNodeSampler()

    outlier_sampler = mh.OutlierSampler()

    tree_prior_dist = FSCRPDistribution(1.0)

    kernel = SemiAdaptedKernel(tree_prior_dist)

    pg_sampler = ParticleGibbsTreeSampler(
        kernel, num_particles=10, outlier_proposal_prob=0.0, propose_roots=True, resample_threshold=0.5
    )

    print('Starting sampling')

    num_iters = int(10000)

    trace = []

    for i in range(num_iters):
        tree = pg_sampler.sample_tree(tree)

#         for _ in range(5):
#             tree = mh_sampler.sample_tree(tree)
# 
#             tree = mh_sampler2.sample_tree(tree)
#
#         for _ in range(100):
#             tree = outlier_sampler.sample_tree(tree)

        tree.relabel_nodes()

#         tree = outlier_node_sampler.sample_tree(tree)

        node_sizes = []

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            node_sizes.append(len(node_data))

        tree_prior_dist.alpha = conc_sampler.sample(tree_prior_dist.alpha, len(tree.nodes), sum(node_sizes))

        if i % 1 == 0:
            pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
            print()
            print(i, tree_prior_dist.alpha)
            print(pred_labels)
            print(homogeneity_completeness_v_measure(labels, pred_labels), len(tree.nodes))
            print(tree.log_p_one)
            print(tree.log_p_one + RootPermutationDistribution.log_pdf(tree))
            print(nx.is_isomorphic(tree._graph, true_graph))
            print(tree.roots)
            print(tree.graph.edges)
            print()

            for node in tree.nodes:
                print(node, [(x + 1) / 101 for x in np.argmax(tree.graph.nodes[node]['log_R'], axis=1)])

            if i >= min((num_iters / 2), 1000):
                trace.append(tree)

    consensus_tree = get_consensus_tree(trace)

    print(consensus_tree.edges())

    for node in consensus_tree.nodes():
        print(node, consensus_tree.nodes[node]['data_points'])

    print()


def init_tree(data):
    kernel = SemiAdaptedKernel(1.0, data[0].shape, 0.1)

    tree = Tree.get_single_node_tree(data)

#     return tree

    mh_sampler = mh.PruneRegraphSampler()

    mh_sampler2 = mh.NodeSwap()

    outlier_sampler = mh.OutlierSampler()

    for i in range(0, 0):
        print(i)
        data_sigma = RootPermutationDistribution.sample(tree)

        smc_sampler = SMCSampler(data_sigma, kernel, num_particles=20, resample_threshold=0.5)

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights)

        tree = swarm.particles[idx].tree

        for _ in range(5):
            tree = mh_sampler.sample_tree(tree)
#
            tree = mh_sampler2.sample_tree(tree)

        for _ in range(100):
            tree = outlier_sampler.sample_tree(tree)

    return tree


if __name__ == "__main__":
    #     import line_profiler
    #
    #     import phyclone.smc.kernels.semi_adapted
    #     import phyclone.smc.samplers
    #     import phyclone.tree
    #
    #     profiler = line_profiler.LineProfiler(
    #         ParticleGibbsSubtreeSampler.sample_swarm,
    #         ParticleGibbsSubtreeSampler.sample_tree,
    #         phyclone.smc.samplers.ConditionalSMCSampler.sample,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedKernel.propose_particle,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedKernel.create_particle,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedProposalDistribution._init_dist,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedProposalDistribution.sample,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedProposalDistribution._propose_existing_node,
    #         phyclone.smc.kernels.semi_adapted.SemiAdaptedProposalDistribution._propose_new_node,
    # #         phyclone.tree.Tree.copy,
    # #         phyclone.tree.Tree.data_marginal_log_likelihood
    #
    #     )
    #
    #     profiler.run("main()")
    #
    #     profiler.print_stats()
    main()
