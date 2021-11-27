"""
Created on 2012-02-08

@author: Andrew Roth
"""
import Bio.Phylo
import gzip
import numpy as np
import pandas as pd
import pickle
import random

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.map import get_map_node_ccfs
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsSubtreeSampler, ParticleGibbsTreeSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution

import phyclone.data.pyclone
import phyclone.math_utils


def write_map_results(in_file, out_table_file, out_tree_file):
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    map_iter = 0

    map_val = float("-inf")

    for i, x in enumerate(results["trace"]):
        if x["log_p"] > map_val:
            map_iter = i

            map_val = x["log_p"]

    data = results["data"]

    tree = Tree.from_dict(data, results["trace"][map_iter]["tree"])

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    table.to_csv(out_table_file, index=False, sep="\t")

    Bio.Phylo.write(get_bp_tree_from_graph(tree.graph), out_tree_file, "newick", plain=True)


def write_consensus_results(in_file, out_table_file, out_tree_file):
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    data = results["data"]

    trees = [Tree.from_dict(data, x["tree"]) for x in results["trace"]]

    graph = phyclone.consensus.get_consensus_tree(trees, data=data)

    tree = get_tree_from_consensus_graph(data, graph)

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    table = pd.DataFrame(table)

    table.to_csv(out_table_file, index=False, sep="\t")

    Bio.Phylo.write(get_bp_tree_from_graph(tree.graph), out_tree_file, "newick", plain=True)


def get_clades(tree, source=None):
    if source is None:
        roots = []

        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                roots.append(node)

        children = []
        for node in roots:
            children.append(get_clades(tree, source=node))

        clades = Bio.Phylo.BaseTree.Clade(name="root", clades=children)

    else:
        children = []

        for child in tree.successors(source):
            children.append(get_clades(tree, source=child))

        clades = Bio.Phylo.BaseTree.Clade(name=str(source), clades=children)

    return clades


def get_bp_tree_from_graph(tree):
    return Bio.Phylo.BaseTree.Tree(root=get_clades(tree), rooted=True)


def get_tree_from_consensus_graph(data, graph):
    labels = {}

    for node in graph.nodes:
        for idx in graph.nodes[node]["idxs"]:
            labels[idx] = node

    for x in data:
        if x.idx not in labels:
            labels[x.idx] = -1

    graph = graph.copy()

    nodes = list(graph.nodes)

    for node in nodes:
        if len(list(graph.predecessors(node))) == 0:
            graph.add_edge("root", node)

    tree = Tree.from_dict(data, {"graph": graph, "labels": labels})

    tree.update()

    return tree


def get_clone_table(data, samples, tree, clusters=None):
    labels = get_labels_table(data, tree, clusters=clusters)

    ccfs = get_map_node_ccfs(tree)

    table = []

    for _, row in labels.iterrows():
        for i, sample_id in enumerate(samples):
            new_row = row.copy()

            new_row["sample_id"] = sample_id

            if new_row["clone_id"] in ccfs:
                new_row["ccf"] = ccfs[new_row["clone_id"]][i]

            else:
                new_row["ccf"] = -1

            table.append(new_row)

    return pd.DataFrame(table)


def get_labels_table(data, tree, clusters=None):
    df = []

    clone_muts = set()

    if clusters is None:
        for idx in tree.labels:
            df.append({
                "mutation_id": data[idx].name,
                "clone_id": tree.labels[idx],
            })

            clone_muts.add(data[idx].name)

        for x in data:
            if x.name not in clone_muts:
                df.append({
                    "mutation_id": x.name,
                    "clone_id": -1
                })

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id"])

        df = df.sort_values(by=["clone_id", "mutation_id"])

    else:
        for idx in tree.labels:
            muts = clusters[clusters["cluster_id"] == int(data[idx].name)]["mutation_id"]

            for mut in muts:
                df.append({
                    "mutation_id": mut,
                    "clone_id": tree.labels[idx],
                    "cluster_id": int(data[idx].name)
                })

                clone_muts.add(mut)

        clusters = clusters.set_index("mutation_id")

        for mut in clusters.index.values:
            if mut not in clone_muts:
                df.append({
                    "mutation_id": mut,
                    "clone_id": -1,
                    "cluster_id": clusters.loc[mut].values[0]
                })

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id", "cluster_id"])

        df = df.sort_values(by=["clone_id", "cluster_id", "mutation_id"])

    return df


def run(
        in_file,
        out_file,
        burnin=100,
        cluster_file=None,
        concentration_value=1.0,
        density="beta-binomial",
        grid_size=101,
        num_iters=1000,
        num_particles=20,
        outlier_prob=0,
        precision=1.0,
        print_freq=100,
        proposal="fully-adapted",
        resample_threshold=0.5,
        seed=None,
        subtree_update_prob=0,
        thin=1):

    if seed is not None:
        np.random.seed(seed)

        random.seed(seed)

    data, samples = phyclone.data.pyclone.load_data(
        in_file, cluster_file=cluster_file, density=density, grid_size=grid_size, outlier_prob=outlier_prob, precision=precision
    )

    tree_dist = TreeJointDistribution(FSCRPDistribution(concentration_value))

    if outlier_prob > 0:
        outlier_proposal_prob = 0.1

    else:
        outlier_proposal_prob = 0

    if proposal == "bootstrap":
        kernel_cls = BootstrapKernel

    elif proposal == "fully-adapted":
        kernel_cls = FullyAdaptedKernel

    elif proposal == "semi-adapted":
        kernel_cls = SemiAdaptedKernel

    kernel = kernel_cls(tree_dist, outlier_proposal_prob=outlier_proposal_prob)

    dp_sampler = DataPointSampler(tree_dist, outliers=(outlier_prob > 0))

    prg_sampler = PruneRegraphSampler(tree_dist)

    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

    # =========================================================================
    # Burnin
    # =========================================================================
    smc_sampler = UnconditionalSMCSampler(
        kernel, num_particles=num_particles, resample_threshold=resample_threshold
    )

    tree = Tree.get_single_node_tree(data)

    print("#" * 100)
    print("Burnin")
    print("#" * 100)

    for i in range(-burnin, 0):
        if (burnin + i) % print_freq == 0:
            print_stats(i, tree, tree_dist)

        tree = smc_sampler.sample_tree(tree)

        tree = dp_sampler.sample_tree(tree)

        tree = prg_sampler.sample_tree(tree)

        tree.relabel_nodes()

    # =========================================================================
    # Main sampler
    # =========================================================================
    tree_sampler = ParticleGibbsTreeSampler(
        kernel, num_particles=20, resample_threshold=0.5
    )

    subtree_sampler = ParticleGibbsSubtreeSampler(
        kernel, num_particles=20, resample_threshold=0.5
    )

    print()
    print("#" * 100)
    print("Post-burnin")
    print("#" * 100)
    print()

    trace = []

    for i in range(num_iters):
        if i % print_freq == 0:
            print_stats(i, tree, tree_dist)

        if random.random() < subtree_update_prob:
            tree = subtree_sampler.sample_tree(tree)

        else:
            tree = tree_sampler.sample_tree(tree)

        for _ in range(5):
            tree = prg_sampler.sample_tree(tree)

            tree = dp_sampler.sample_tree(tree)

        tree.relabel_nodes()

        node_sizes = []

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            node_sizes.append(len(node_data))

        tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(tree.nodes), sum(node_sizes))

        if i % thin == 0:
            trace.append({
                "iter": i,
                "alpha": tree_dist.prior.alpha,
                "log_p": tree_dist.log_p_one(tree),
                "tree": tree.to_dict()
            })

    results = {"data": data, "samples": samples, "trace": trace}

    if cluster_file is not None:
        results["clusters"] = pd.read_csv(cluster_file, sep="\t")[["mutation_id", "cluster_id"]].drop_duplicates()

    with gzip.GzipFile(out_file, mode="wb") as fh:
        pickle.dump(results, fh)


def print_stats(iter_id, tree, tree_dist):
    print(iter_id, tree_dist.prior.alpha, tree_dist.log_p_one(tree),
          len(tree.nodes), len(tree.outliers), len(tree.roots))


class UnconditionalSMCSampler(object):

    def __init__(self, kernel, num_particles=20, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

    def sample_tree(self, tree):
        data_sigma = RootPermutationDistribution.sample(tree)

        smc_sampler = SMCSampler(
            data_sigma, self.kernel, num_particles=self.num_particles, resample_threshold=self.resample_threshold
        )

        swarm = smc_sampler.sample()

        idx = phyclone.math_utils.discrete_rvs(swarm.weights)

        return swarm.particles[idx].tree