'''
Created on 2012-02-08

@author: Andrew Roth
'''
import Bio.Phylo
import numpy as np
import pandas as pd
import random

from phyclone.mcmc.metropolis_hastings import NodeSwap, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsSubtreeSampler
from phyclone.smc.kernels import SemiAdaptedKernel
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.tree import FSCRPDistribution, Tree

import phyclone.data.pyclone
import phyclone.math_utils
import phyclone.trace


def post_process(data_file, trace_file, out_table_file, out_tree_file):
    data = phyclone.data.pyclone.load_data(data_file)

    trace = phyclone.trace.Trace(trace_file, mode='r')

    trees = trace.load(data)

    print(len(trees))

    graph = phyclone.consensus.get_consensus_tree(trees, data=data)

    table = get_clone_table(data, graph)

    table.to_csv(out_table_file, index=False, sep='\t')

    Bio.Phylo.write(get_tree(graph), out_tree_file, 'newick', plain=True)


def get_clades(tree, source=None):
    if source is None:
        roots = []

        for node in tree.nodes:
            if tree.in_degree(node) == 0:
                roots.append(node)

        children = []
        for node in roots:
            children.append(get_clades(tree, source=node))

        clades = Bio.Phylo.BaseTree.Clade(name='root', clades=children)

    else:
        children = []

        for child in tree.successors(source):
            children.append(get_clades(tree, source=child))

        clades = Bio.Phylo.BaseTree.Clade(name=str(source), clades=children)

    return clades


def get_tree(tree):
    return Bio.Phylo.BaseTree.Tree(root=get_clades(tree), rooted=True)


def get_clone_table(data, graph):
    df = []

    for node in graph.nodes:
        for idx in graph.nodes[node]['idxs']:
            df.append({
                'mutation_id': data[idx].name,
                'clone_id': node,
            })

    df = pd.DataFrame(df, columns=['mutation_id', 'clone_id'])

    df = df.sort_values(by=['clone_id', 'mutation_id'])

    return df


def run(
        in_file,
        trace_file,
        burnin=100,
        concentration_value=1.0,
        density='beta-binomial',
        grid_size=None,
        num_iters=1000,
        precision=1.0,
        seed=None):

    if seed is not None:
        np.random.seed(seed)

        random.seed(seed)

    data = phyclone.data.pyclone.load_data(
        in_file, density=density, grid_size=grid_size, outlier_prob=0, precision=precision
    )

    tree_prior_dist = FSCRPDistribution(concentration_value)

    kernel = SemiAdaptedKernel(tree_prior_dist, outlier_proposal_prob=0.0)

    #=========================================================================
    # Burnin
    #=========================================================================
    smc_sampler = UnconditionalSMCSampler(
        kernel, num_particles=20, resample_threshold=0.5
    )

    sampler = MixedSampler(smc_sampler)

    tree = Tree.get_single_node_tree(data)

    for i in range(-burnin, 0):
        print(i)

        tree = sampler.sample_tree(tree)

    #=========================================================================
    # Main sampler
    #=========================================================================
    smc_sampler = ParticleGibbsSubtreeSampler(
        kernel, num_particles=20, propose_roots=True, resample_threshold=0.5
    )

    sampler = MixedSampler(smc_sampler)

    trace = phyclone.trace.Trace(trace_file, mode='w')

    for i in range(0, num_iters):
        print(i)

        tree = sampler.sample_tree(tree)

        trace.update(tree)

    trace.close()


class MixedSampler(object):
    def __init__(self, smc_sampler):
        self.smc_sampler = smc_sampler

        self.prg_sampler = PruneRegraphSampler()

        self.nsw_sampler = NodeSwap()

    def sample_tree(self, tree):
        tree = self.smc_sampler.sample_tree(tree)

        for _ in range(5):
            tree = self.prg_sampler.sample_tree(tree)

            tree = self.nsw_sampler.sample_tree(tree)

        return tree


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
