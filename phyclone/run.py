"""
Created on 2012-02-08

@author: Andrew Roth
"""
import numpy as np
from numba import set_num_threads
from dataclasses import dataclass

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsSubtreeSampler, ParticleGibbsTreeSampler
from phyclone.process_trace import _create_main_run_output
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import SMCSampler
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils import Timer

from phyclone.data.pyclone import load_data
from phyclone.math_utils import discrete_rvs


def run(
        in_file,
        out_file,
        burnin=100,
        cluster_file=None,
        concentration_value=1.0,
        concentration_update=True,
        density="beta-binomial",
        grid_size=101,
        max_time=float("inf"),
        num_iters=1000,
        num_particles=20,
        num_samples_data_point=1,
        num_samples_prune_regraph=1,
        outlier_prob=0,
        precision=1.0,
        print_freq=100,
        proposal="fully-adapted",
        resample_threshold=0.5,
        seed=None,
        subtree_update_prob=0,
        thin=1,
        num_threads=1,
        mitochondrial=False):

    rng = instantiate_and_seed_RNG(seed)

    set_num_threads(num_threads)

    data, samples = load_data(
        in_file, cluster_file=cluster_file, density=density, grid_size=grid_size, outlier_prob=outlier_prob,
        precision=precision, mitochondrial=mitochondrial)

    tree_dist = TreeJointDistribution(FSCRPDistribution(concentration_value))

    kernel = setup_kernel(outlier_prob, proposal, rng, tree_dist)

    samplers = setup_samplers(kernel,
                              num_particles,
                              outlier_prob,
                              resample_threshold,
                              rng,
                              tree_dist)

    tree = Tree.get_single_node_tree(data)

    timer = Timer()

    # =========================================================================
    # Burnin
    # =========================================================================
    tree = _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer,
                       tree, tree_dist)

    # =========================================================================
    # Main sampler
    # =========================================================================

    trace = setup_trace(timer, tree, tree_dist)

    results = _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                                num_samples_prune_regraph, print_freq, rng, samplers, samples, subtree_update_prob,
                                thin, timer, trace, tree, tree_dist)

    _create_main_run_output(cluster_file, out_file, results)


def _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                      num_samples_prune_regraph, print_freq, rng, samplers, samples, subtree_update_prob, thin, timer,
                      trace, tree, tree_dist):
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    subtree_sampler = samplers.subtree_sampler
    tree_sampler = samplers.tree_sampler
    conc_sampler = samplers.conc_sampler

    for i in range(num_iters):
        with timer:
            if i % print_freq == 0:
                print_stats(i, tree, tree_dist)

            if rng.random() < subtree_update_prob:
                tree = subtree_sampler.sample_tree(tree)
            else:
                tree = tree_sampler.sample_tree(tree)

            for _ in range(num_samples_data_point):
                tree = dp_sampler.sample_tree(tree)

            for _ in range(num_samples_prune_regraph):
                tree = prg_sampler.sample_tree(tree)

            tree.relabel_nodes()

            if concentration_update:
                update_concentration_value(conc_sampler, tree, tree_dist)

            if i % thin == 0:
                append_to_trace(i, timer, trace, tree, tree_dist)

            if timer.elapsed >= max_time:
                break
    results = {"data": data, "samples": samples, "trace": trace}
    return results


def append_to_trace(i, timer, trace, tree, tree_dist):
    trace.append({
        "iter": i,
        "time": timer.elapsed,
        "alpha": tree_dist.prior.alpha,
        "log_p": tree_dist.log_p_one(tree),
        "tree": tree.to_dict()
    })


def update_concentration_value(conc_sampler, tree, tree_dist):
    node_sizes = []
    for node, node_data in tree.node_data.items():
        if node == -1:
            continue

        node_sizes.append(len(node_data))

    tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(tree.nodes), sum(node_sizes))


def setup_trace(timer, tree, tree_dist):
    trace = []
    append_to_trace(0, timer, trace, tree, tree_dist)
    return trace


def _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer, tree,
                tree_dist):
    burnin_sampler = samplers.burnin_sampler
    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    if burnin > 0:
        print("#" * 100)
        print("Burnin")
        print("#" * 100)

        for i in range(burnin):
            with timer:
                if i % print_freq == 0:
                    print_stats(i, tree, tree_dist)

                tree = burnin_sampler.sample_tree(tree)

                for _ in range(num_samples_data_point):
                    tree = dp_sampler.sample_tree(tree)

                for _ in range(num_samples_prune_regraph):
                    tree = prg_sampler.sample_tree(tree)

                tree.relabel_nodes()

                if timer.elapsed > max_time:
                    break

    print()
    print("#" * 100)
    print("Post-burnin")
    print("#" * 100)
    print()

    return tree


class UnconditionalSMCSampler(object):

    def __init__(self, kernel, num_particles=20, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self._rng = kernel.rng

    def sample_tree(self, tree):
        data_sigma = RootPermutationDistribution.sample(tree, self._rng)

        smc_sampler = SMCSampler(
            data_sigma, self.kernel, num_particles=self.num_particles, resample_threshold=self.resample_threshold
        )

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights, self._rng)

        return swarm.particles[idx].tree


@dataclass
class SamplersHolder:
    dp_sampler: DataPointSampler
    prg_sampler: PruneRegraphSampler
    conc_sampler: GammaPriorConcentrationSampler
    burnin_sampler: UnconditionalSMCSampler
    tree_sampler: ParticleGibbsTreeSampler
    subtree_sampler: ParticleGibbsSubtreeSampler


def setup_samplers(kernel, num_particles, outlier_prob, resample_threshold, rng, tree_dist):
    dp_sampler = DataPointSampler(tree_dist, rng, outliers=(outlier_prob > 0))
    prg_sampler = PruneRegraphSampler(tree_dist, rng)
    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)
    burn_in_particles = int(max(1, np.rint(num_particles / 2)))
    burnin_sampler = UnconditionalSMCSampler(
        kernel, num_particles=burn_in_particles, resample_threshold=resample_threshold
    )
    tree_sampler = ParticleGibbsTreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    subtree_sampler = ParticleGibbsSubtreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    return SamplersHolder(dp_sampler,
                          prg_sampler,
                          conc_sampler,
                          burnin_sampler,
                          tree_sampler,
                          subtree_sampler)


def setup_kernel(outlier_prob, proposal, rng, tree_dist):
    if outlier_prob > 0:
        outlier_proposal_prob = 0.1
    else:
        outlier_proposal_prob = 0
    kernel_cls = FullyAdaptedKernel
    if proposal == "bootstrap":
        kernel_cls = BootstrapKernel
    elif proposal == "fully-adapted":
        kernel_cls = FullyAdaptedKernel
    elif proposal == "semi-adapted":
        kernel_cls = SemiAdaptedKernel
    kernel = kernel_cls(tree_dist, rng, outlier_proposal_prob=outlier_proposal_prob)
    return kernel


def instantiate_and_seed_RNG(seed):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng


def print_stats(iter_id, tree, tree_dist):
    string_template = 'iter: {}, alpha: {}, log_p: {}, num_nodes: {}, num_outliers: {}, num_roots: {}'
    print(string_template.format(iter_id, round(tree_dist.prior.alpha, 3), round(tree_dist.log_p_one(tree), 3),
                                 len(tree.nodes), len(tree.outliers), len(tree.roots)))
