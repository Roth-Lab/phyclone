"""
Created on 2012-02-08

@author: Andrew Roth
"""
import numpy as np
from dataclasses import dataclass

from phyclone.mcmc.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.gibbs_mh import DataPointSampler, PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.process_trace import create_main_run_output
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.smc.samplers import UnconditionalSMCSampler
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.utils import Timer, read_pickle, save_numpy_rng
from phyclone.data.pyclone import load_data
from numba import set_num_threads
from phyclone.utils.dev import clear_proposal_dist_caches
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        proposal="semi-adapted",
        resample_threshold=0.5,
        seed=None,
        thin=1,
        num_threads=1,
        rng_pickle=None,
        save_rng=True):
    rng_main = instantiate_and_seed_RNG(seed, rng_pickle)

    if save_rng:
        save_numpy_rng(out_file, rng_main)

    data, samples = load_data(
        in_file, cluster_file=cluster_file, density=density, grid_size=grid_size, outlier_prob=outlier_prob,
        precision=precision)

    results = {}

    if num_threads == 1:
        results[0] = phyclone_go(burnin, concentration_update, concentration_value, data, max_time, num_iters,
                                 num_particles, num_samples_data_point, num_samples_prune_regraph, outlier_prob,
                                 print_freq, proposal, resample_threshold, rng_main, samples, thin, 0)

        print("Finished chain", 0)

    else:

        rng_list = rng_main.spawn(num_threads)

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            chain_results = [pool.submit(phyclone_go, burnin, concentration_update, concentration_value,
                                         data, max_time, num_iters,
                                         num_particles, num_samples_data_point,
                                         num_samples_prune_regraph, outlier_prob,
                                         print_freq, proposal, resample_threshold,
                                         rng, samples, thin, chain_num) for chain_num, rng in enumerate(rng_list)]

            for future in as_completed(chain_results):
                exception = future.exception()
                if exception is not None:
                    raise exception
                else:
                    result = future.result()
                    res_chain = result["chain_num"]
                    results[res_chain] = result
                    print("Finished chain", res_chain)

    create_main_run_output(cluster_file, out_file, results)


def phyclone_go(burnin, concentration_update, concentration_value, data, max_time, num_iters, num_particles,
                num_samples_data_point, num_samples_prune_regraph, outlier_prob, print_freq, proposal,
                resample_threshold, rng, samples, thin, chain_num):
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
    tree = _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer,
                       tree, tree_dist, chain_num)
    results = _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                                num_samples_prune_regraph, print_freq, samplers, samples, thin, timer, tree, tree_dist,
                                chain_num)
    return results


def set_numba_run_threads(num_threads, samples):
    # guarding against bad user inputs
    num_threads = max(1, num_threads)
    # don't use more threads than there are samples, numba goes slower
    threads_to_use = min(num_threads, len(samples))
    set_num_threads(threads_to_use)


def _run_main_sampler(concentration_update, data, max_time, num_iters, num_samples_data_point,
                      num_samples_prune_regraph, print_freq, samplers, samples, thin, timer, tree, tree_dist,
                      chain_num):
    trace = setup_trace(timer, tree, tree_dist)

    dp_sampler = samplers.dp_sampler
    prg_sampler = samplers.prg_sampler
    tree_sampler = samplers.tree_sampler
    conc_sampler = samplers.conc_sampler

    for i in range(num_iters):
        with timer:
            if i % print_freq == 0:
                print_stats(i, tree, tree_dist, chain_num)

            clear_proposal_dist_caches()

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
    results = {"data": data, "samples": samples, "trace": trace, "chain_num": chain_num}
    return results


def append_to_trace(i, timer, trace, tree, tree_dist):
    trace.append({
        "iter": i,
        "time": timer.elapsed,
        "alpha": tree_dist.prior.alpha,
        "log_p_one": tree_dist.log_p_one(tree),
        "tree": tree.to_dict()
    })


def update_concentration_value(conc_sampler, tree, tree_dist):
    node_sizes = []
    for node, node_data in tree.node_data.items():
        if node == -1:
            continue

        node_sizes.append(len(node_data))

    tree_dist.prior.alpha = conc_sampler.sample(tree_dist.prior.alpha, len(node_sizes), sum(node_sizes))


def setup_trace(timer, tree, tree_dist):
    trace = []
    append_to_trace(0, timer, trace, tree, tree_dist)
    return trace


def _run_burnin(burnin, max_time, num_samples_data_point, num_samples_prune_regraph, print_freq, samplers, timer, tree,
                tree_dist, chain_num):
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
                    print_stats(i, tree, tree_dist, chain_num)

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


@dataclass
class SamplersHolder:
    dp_sampler: DataPointSampler
    prg_sampler: PruneRegraphSampler
    conc_sampler: GammaPriorConcentrationSampler
    burnin_sampler: UnconditionalSMCSampler
    tree_sampler: ParticleGibbsTreeSampler


def setup_samplers(kernel, num_particles, outlier_prob, resample_threshold, rng, tree_dist):
    dp_sampler = DataPointSampler(tree_dist, rng, outliers=(outlier_prob > 0))
    prg_sampler = PruneRegraphSampler(tree_dist, rng)
    conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)
    burnin_sampler = UnconditionalSMCSampler(
        kernel, num_particles=num_particles, resample_threshold=resample_threshold
    )
    tree_sampler = ParticleGibbsTreeSampler(
        kernel, rng, num_particles=num_particles, resample_threshold=resample_threshold
    )
    return SamplersHolder(dp_sampler,
                          prg_sampler,
                          conc_sampler,
                          burnin_sampler,
                          tree_sampler)


def setup_kernel(outlier_prob, proposal, rng, tree_dist):
    if outlier_prob > 0:
        outlier_proposal_prob = 0.1
    else:
        outlier_proposal_prob = 0
    kernel_cls = SemiAdaptedKernel
    if proposal == "bootstrap":
        kernel_cls = BootstrapKernel
    elif proposal == "fully-adapted":
        kernel_cls = FullyAdaptedKernel
    elif proposal == "semi-adapted":
        kernel_cls = SemiAdaptedKernel

    kernel = kernel_cls(tree_dist, rng, outlier_proposal_prob=outlier_proposal_prob)
    return kernel


def instantiate_and_seed_RNG(seed, rng_pickle):
    if (seed is not None) and (rng_pickle is None):
        rng = np.random.default_rng(seed)
    elif rng_pickle is not None:
        loaded = read_pickle(rng_pickle)
        rng = np.random.default_rng(loaded)
    else:
        rng = np.random.default_rng()
    return rng


def print_stats(iter_id, tree, tree_dist, chain_num):
    string_template = 'chain: {} || iter: {}, alpha: {}, log_p: {}, num_nodes: {}, num_outliers: {}, num_roots: {}'
    print(string_template.format(chain_num, iter_id, round(tree_dist.prior.alpha, 3),
                                 round(tree_dist.log_p_one(tree), 3),
                                 tree.get_number_of_nodes(), len(tree.outliers), tree.get_number_of_children("root")))
