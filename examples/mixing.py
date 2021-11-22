from sklearn.metrics import homogeneity_completeness_v_measure

import networkx as nx
import numpy as np
import pandas as pd
import random

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.consensus import get_consensus_tree
from phyclone.data.pyclone import load_data
from phyclone.math_utils import discrete_rvs
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler, ParticleGibbsSubtreeSampler
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.smc.utils import RootPermutationDistribution

import phyclone.mcmc.gibbs_mh as mh

from phyclone.map import get_map_node_ccfs


def main():
    outlier_prob = 0
    subtree_prob = 0.5
    kernel_type = "fully-adapted"
    file_name = "data/mixing_small.tsv"
#     file_name = "data/mixing.tsv"
#     cluster_file = "data/mixing_small_clusters.tsv"
#     cluster_file = "data/mixing_clusters.tsv"
    cluster_file = None

    true_labels = load_true_labels(file_name)
    
    true_tree = load_true_tree()
    
    data = load_data(
        file_name,
        cluster_file=cluster_file,
        density="beta-binomial",
        grid_size=101,
        precision=50,
        outlier_prob=outlier_prob
    )

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))
    
    mh_sampler_1 = mh.PruneRegraphSampler(tree_dist)

    mh_sampler_2 = mh.DataPointSampler(tree_dist, outliers=(outlier_prob > 0))
    
    if kernel_type == "bootstrap":
        kernel_cls = BootstrapKernel
        
    elif kernel_type == "semi-adapted":
        kernel_cls = SemiAdaptedKernel    
    
    elif kernel_type == "fully-adapted": 
        kernel_cls = FullyAdaptedKernel
    
    else:
        raise Exception("Unknown kernel type: {}".format(kernel_type))
    
    if outlier_prob > 0:
        outlier_proposal_prob = 0.1
    
    else:
        outlier_proposal_prob = 0

    kernel = kernel_cls(tree_dist, outlier_proposal_prob=outlier_proposal_prob)

    tree = init_tree(data, kernel=None, mcmc_samplers=[mh_sampler_1, mh_sampler_2])
    
    subtree_sampler = ParticleGibbsSubtreeSampler(
        kernel, num_particles=20, resample_threshold=0.5
    )        

    tree_sampler = ParticleGibbsTreeSampler(
        kernel, num_particles=20, resample_threshold=0.5
    )

    print("Starting sampling")

    num_iters = int(10000)

    trace = []

    for i in range(num_iters):
        if random.random() < subtree_prob:
            tree = subtree_sampler.sample_tree(tree)
        
        else:
            tree = tree_sampler.sample_tree(tree)

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

        if i % 10 == 0:
            pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
            print()
            print(i, tree_dist.prior.alpha)
            print(pred_labels)
            if cluster_file is None:
                print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
            print(tree_dist.log_p_one(tree))
            print(nx.is_isomorphic(tree.graph, true_tree))
            print(tree.roots)
            print(tree.graph.edges)
            print()
            
            ccfs = get_map_node_ccfs(tree)
            print("MAP CCF")
            for n in sorted(ccfs):
                print(n, ccfs[n])
            print()
            
            print("Marginal CCF")
            for node in sorted(tree.nodes):
                print(node, [x / 100 for x in np.argmax(tree.graph.nodes[node]["log_R"], axis=1)])
            print()

            if i >= min((num_iters / 2), 1000):
                trace.append(tree)
            
            for node in sorted(tree.node_data):
                print(node, [x.name for x in tree.node_data[node]])

    consensus_tree = get_consensus_tree(trace)

    print(consensus_tree.edges())

    for node in consensus_tree.nodes():
        print(node, consensus_tree.nodes[node]["data_points"])

    print()


def init_tree(data, kernel=None, mcmc_samplers=[]):
    tree = Tree.get_single_node_tree(data)
    
    if kernel is None:
        return tree

    for i in range(1):
        print(i)
        data_sigma = RootPermutationDistribution.sample(tree)

        smc_sampler = SMCSampler(data_sigma, kernel, num_particles=20, resample_threshold=0.5)

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights)

        tree = swarm.particles[idx].tree

        for _ in range(5):
            for sampler in mcmc_samplers:
                tree = sampler.sample_tree(tree)

    return tree


def load_true_labels(file_name):
    df = pd.read_csv(file_name, sep="\t")
    
    cluster_names = list(df["variant_cases"].unique())
    
    df = df[["mutation_id", "variant_cases"]].drop_duplicates()
    
    return df["variant_cases"].apply(lambda x: cluster_names.index(x))


def load_true_tree():
    branches = [
        ("NA12156,NA12878,NA18507,NA19240", "NA12878,NA18507,NA19240"),
        ("NA12156,NA12878,NA18507,NA19240", "NA12156"),
        ("NA12878,NA18507,NA19240", "NA18507,NA19240"),
        ("NA12878,NA18507,NA19240", "NA12878"),
        ("NA18507,NA19240", "NA18507"),
        ("NA18507,NA19240", "NA19240")
    ]
    
    graph = nx.DiGraph()
    
    for u, v in branches:
        graph.add_edge(u, v)
    
    return graph


if __name__ == "__main__":
    main()
