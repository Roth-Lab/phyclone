from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np

from phyclone.mcmc.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.map import get_map_node_ccfs
from phyclone.utils.math import discrete_rvs
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler, ParticleGibbsSubtreeSampler
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.smc.utils import RootPermutationDistribution
from examples.metrics import partition_metric

import phyclone.mcmc.gibbs_mh as mh

from toy_data import load_test_data
from phyclone.run import instantiate_and_seed_RNG


def main(seed=1234):
    rng = instantiate_and_seed_RNG(seed, None)
    subtree = True
    kernel_type = "fully-adapted"
    
    data, true_tree = load_test_data(rng, cluster_size=5, depth=int(5e1), outlier_size=1, single_sample=False)

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)

    tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))
    
    mh_sampler_1 = mh.PruneRegraphSampler(tree_dist, rng=rng)

    mh_sampler_2 = mh.DataPointSampler(tree_dist, outliers=True, rng=rng)
    
    if kernel_type == "bootstrap":
        kernel_cls = BootstrapKernel
        
    elif kernel_type == "semi-adapted":
        kernel_cls = SemiAdaptedKernel    
    
    elif kernel_type == "fully-adapted": 
        kernel_cls = FullyAdaptedKernel
    
    else:
        raise Exception("Unknown kernel type: {}".format(kernel_type))

    kernel = kernel_cls(tree_dist, outlier_proposal_prob=0.1, rng=rng)

    tree = init_tree(data, kernel=None, mcmc_samplers=[mh_sampler_1, mh_sampler_2], rng=rng)
    
    if subtree:
        pg_sampler = ParticleGibbsSubtreeSampler(
            kernel, rng=rng, num_particles=20, resample_threshold=0.5
        )        
       
    else:
        pg_sampler = ParticleGibbsTreeSampler(
            kernel, rng=rng, num_particles=20, resample_threshold=0.5
        )

    print("Starting sampling")

    num_iters = int(10000)

    trace = []

    for i in range(num_iters):
        tree = pg_sampler.sample_tree(tree)

        for _ in range(5):
            tree = mh_sampler_1.sample_tree(tree)
        
        tree = mh_sampler_2.sample_tree(tree)

        tree.relabel_nodes()

        node_sizes = []

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            node_sizes.append(len(node_data))

        tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(tree.nodes), sum(node_sizes))

        if i % 1 == 0:
            pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
            true_labels = [true_tree.labels[x] for x in sorted(true_tree.labels)]
            print()
            print(i, tree_dist.prior.alpha)
            print(pred_labels)
            print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
            print(tree_dist.log_p_one(tree))
            print(nx.is_isomorphic(tree.graph, true_tree.graph))
            print(partition_metric(tree, true_tree))
            print(tree.roots)
            print(tree.graph.edges)
            
            ccfs = get_map_node_ccfs(tree)
            print("MAP CCF")
            for n in sorted(ccfs):
                print(n, ccfs[n])
            print()
            
            print("Marginal CCF")
            for node in sorted(tree.nodes):
                print(node, [x / 100 for x in np.argmax(tree.graph.nodes[node]["log_R"], axis=1)])
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


def init_tree(data, rng, kernel=None, mcmc_samplers=[]):
    tree = Tree.get_single_node_tree(data)
    
    if kernel is None:
        return tree

    for i in range(1):
        print(i)
        data_sigma = RootPermutationDistribution.sample(tree, rng=rng)

        smc_sampler = SMCSampler(data_sigma, kernel, num_particles=20, resample_threshold=0.5)

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights, rng=rng)

        tree = swarm.particles[idx].tree

        for _ in range(5):
            for sampler in mcmc_samplers:
                tree = sampler.sample_tree(tree)

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
