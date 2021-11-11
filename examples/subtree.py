from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.math_utils import discrete_rvs
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler, ParticleGibbsSubtreeSampler
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.tree import Tree, FSCRPDistribution
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.metrics import partition_metric

import phyclone.mcmc.metropolis_hastings as mh

from toy_data import load_test_data


def main():
    subtree = True
    kernel_type = "semi-adapted"
    
    data, true_tree = load_test_data(cluster_size=5, depth=int(1e2), outlier_size=1, single_sample=False)

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    mh_sampler = mh.PruneRegraphSampler()

    mh_sampler2 = mh.NodeSwap()
    
    if kernel_type == "bootstrap":
        kernel_cls = BootstrapKernel
        
    elif kernel_type == "semi-adapted":
        kernel_cls = SemiAdaptedKernel    
    
    elif kernel_type == "fully-adapted": 
        kernel_cls = FullyAdaptedKernel
    
    else:
        raise Exception("Unknown kernel type: {}".format(kernel_type))

    outlier_sampler = mh.OutlierSampler()

    tree_prior_dist = FSCRPDistribution(1.0)
    
    kernel = kernel_cls(tree_prior_dist, outlier_proposal_prob=0.1)

    tree = init_tree(data, kernel=kernel)
    
    if subtree:
        pg_sampler = ParticleGibbsTreeSampler(
            kernel, num_particles=20, propose_roots=True, resample_threshold=0.5
        )
       
    else: 
        pg_sampler = ParticleGibbsSubtreeSampler(
            kernel, num_particles=20, propose_roots=True, resample_threshold=0.5
        )

    print("Starting sampling")

    num_iters = int(10000)

    trace = []

    for i in range(num_iters):
        tree = pg_sampler.sample_tree(tree)

        for _ in range(5):
            tree = mh_sampler.sample_tree(tree)

            tree = mh_sampler2.sample_tree(tree)

        for _ in range(len(data)):
            tree = outlier_sampler.sample_tree(tree)

        tree.relabel_nodes()

        node_sizes = []

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            node_sizes.append(len(node_data))

        tree_prior_dist.alpha = conc_sampler.sample(tree_prior_dist.alpha, len(tree.nodes), sum(node_sizes))

        if i % 1 == 0:
            pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
            true_labels = [true_tree.labels[x] for x in sorted(tree.labels)]
            print()
            print(i, tree_prior_dist.alpha)
            print(pred_labels)
            print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
            print(tree.log_p_one)
            print(tree.log_p_one + RootPermutationDistribution.log_pdf(tree))
            print(nx.is_isomorphic(tree.graph, true_tree.graph))
            print(partition_metric(tree, true_tree))
            print(tree.roots)
            print(tree.graph.edges)
            print()

            for node in tree.nodes:
                print(node, [(x + 1) / 101 for x in np.argmax(tree.graph.nodes[node]["log_R"], axis=1)])

            if i >= min((num_iters / 2), 1000):
                trace.append(tree)

    consensus_tree = get_consensus_tree(trace)

    print(consensus_tree.edges())

    for node in consensus_tree.nodes():
        print(node, consensus_tree.nodes[node]["data_points"])

    print()


def init_tree(data, kernel=None):
    tree = Tree.get_single_node_tree(data)
    
    if kernel is None:
        return tree

    mh_sampler = mh.PruneRegraphSampler()

    mh_sampler2 = mh.NodeSwap()

    for i in range(1):
        print(i)
        data_sigma = RootPermutationDistribution.sample(tree)

        smc_sampler = SMCSampler(data_sigma, kernel, num_particles=20, resample_threshold=0.5)

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights)

        tree = swarm.particles[idx].tree

        for _ in range(5):
            tree = mh_sampler.sample_tree(tree)

            tree = mh_sampler2.sample_tree(tree)

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
