import numpy as np
from phyclone.data.pyclone import log_pyclone_binomial_pdf, log_pyclone_beta_binomial_pdf
from scipy.stats import dirichlet
from scipy.special import logsumexp


def run_importance_sampler(num_iters, tree, diri_dist, density, precision, node_post_order, log_p_prior, trial):
    num_samples = tree.graph['num_samples']
    num_nodes = tree.graph['num_nodes']

    trimmed_post_order = tree.graph['node_post_order_stripped']

    weights = np.empty(num_iters, dtype=np.float64)


    for i in range(num_iters):
        if i % 10000 == 0:
            print("trial {}, IS iter {}/{}".format(trial, i, num_iters))

        cell_prev, clonal_prev = sample_clonal_and_cell_prev(num_samples, diri_dist, tree, trimmed_post_order)

        node_llh_arr = compute_node_log_likelihoods(cell_prev, node_post_order, num_nodes, num_samples, tree, density,
                                                    precision)

        loss_ratio = node_llh_arr.sum() + log_p_prior
        weights[i] = loss_ratio

    sum_of_lls = logsumexp(weights)
    avg_llh = sum_of_lls - np.log(len(weights))

    return avg_llh


def get_dirichlet(rng, ones_arr):
    diri = dirichlet(ones_arr, seed=rng)
    return diri


def compute_node_log_likelihoods(cell_prev, node_post_order, num_nodes, num_samples, tree, density, precision):

    node_llh_arr = np.empty((num_samples, num_nodes))
    for node in node_post_order:
        snvs = tree.nodes[node]['snvs']

        node_cell_prev = cell_prev[:, node]
        llh_arr = np.zeros(num_samples)

        for snv in snvs:
            for sample in range(num_samples):
                numba_sample_dp = tree.nodes[node]['converted_sample_dp'][sample][snv]
                mut_ccf = node_cell_prev[sample]
                if density == 'binomial':
                    llh = log_pyclone_binomial_pdf(numba_sample_dp, mut_ccf)
                else:
                    llh = log_pyclone_beta_binomial_pdf(numba_sample_dp, mut_ccf, precision)
                llh_arr[sample] += llh

        node_llh_arr[:, node] = llh_arr
    return node_llh_arr


def sample_clonal_and_cell_prev(num_samples, diri, tree, trimmed_post_order):
    clonal_prev = draw_clonal_prev(num_samples, diri)
    cell_prev = compute_cell_prev_given_clonal_prev(clonal_prev, tree, trimmed_post_order)
    return cell_prev, clonal_prev


def compute_cell_prev_given_clonal_prev(clonal_prev, tree, trimmed_post_order):
    cell_prev = clonal_prev.copy()
    for node in trimmed_post_order:
        children = tree.nodes[node]['children']
        child_prevs = cell_prev[:, children].T
        cell_prev[:, node] += child_prevs.sum(axis=0)
    return cell_prev


def draw_clonal_prev(num_samples, diri):
    return diri.rvs(size=num_samples)
