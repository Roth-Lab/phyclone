import numpy as np
from phyclone.data.pyclone import log_pyclone_binomial_pdf, log_pyclone_beta_binomial_pdf
from scipy.special import logsumexp
from numba import types, njit
from numba.typed import Dict
from numba.experimental import jitclass
from phyclone.data.pyclone import SampleDataPoint
from math import floor


@jitclass
class NumbaNode(object):
    id_num: types.int64
    snvs: types.int64[:]
    children: types.int64[:]
    converted_sample_dp: types.DictType(
        types.int64,
        types.DictType(
            types.int64,
            SampleDataPoint.class_type.instance_type,
        ),
    )

    def __init__(self, id_num, snvs, children, converted_sample_dp):
        self.id_num = id_num
        self.snvs = snvs
        self.children = children
        self.converted_sample_dp = converted_sample_dp

    def get_sample_dp(self, sample, snv):
        return self.converted_sample_dp[sample][snv]


@jitclass
class NumbaTree(object):
    nodes: types.DictType(types.int64, NumbaNode.class_type.instance_type)

    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes[node_id]


def convert_nx_tree_to_nb_tree(nx_tree):

    samp_type = SampleDataPoint.class_type.instance_type

    nx_nodes = np.array(nx_tree.nodes, dtype=np.int64)

    nb_tree_nodes_dict = Dict.empty(types.int64, NumbaNode.class_type.instance_type)

    for node in nx_nodes:
        children = nx_tree.nodes[node]["children"]
        snvs = nx_tree.nodes[node]["snvs"]
        snvs_idx_map = {k: v for v, k in enumerate(snvs)}
        nb_snvs = np.arange(len(snvs_idx_map), dtype=np.int64)
        converted_sample_dp = nx_tree.nodes[node]["converted_sample_dp"]
        nb_cnv_dp = convert_nx_tree_sample_dp_dict_into_nb(converted_sample_dp, samp_type, snvs_idx_map)
        numba_node = NumbaNode(node, nb_snvs, children, nb_cnv_dp)
        nb_tree_nodes_dict[node] = numba_node

    numba_tree = NumbaTree(nb_tree_nodes_dict)
    return numba_tree


def convert_nx_tree_sample_dp_dict_into_nb(converted_sample_dp, samp_type, snvs_idx_map):
    nb_cnv_dp = Dict.empty(types.int64, types.DictType(types.int64, samp_type))
    for k, v in converted_sample_dp.items():
        curr_nb_dict = Dict.empty(types.int64, samp_type)
        for snv, samp_dp in v.items():
            curr_nb_dict[snvs_idx_map[snv]] = samp_dp
        nb_cnv_dp[k] = curr_nb_dict
    return nb_cnv_dp


def run_importance_sampler(num_iters, tree, rng, density, precision, node_post_order, log_p_prior, trial):
    num_samples = int(tree.graph["num_samples"])
    num_nodes = int(tree.graph["num_nodes"])

    trimmed_post_order = tree.graph["node_post_order_stripped"]

    numba_tree = convert_nx_tree_to_nb_tree(tree)

    trimmed_post_order = np.array(trimmed_post_order, dtype=np.int64)

    weights = np.empty(num_iters, dtype=np.float64)

    node_post_order = np.array(node_post_order, dtype=np.int64)

    ones_arr = np.ones(num_nodes)

    num_rounds, round_iters_arr = get_num_rounds_and_round_iters_arr(num_iters)

    w_idx_start = 0

    for i in range(num_rounds):
        round_iters = round_iters_arr[i]
        clonal_prevs_for_round = rng.dirichlet(ones_arr, size=(round_iters, num_samples))

        print("trial {}, IS iter {}/{}".format(trial, w_idx_start, num_iters))

        run_batch_of_IS_iters(
            clonal_prevs_for_round,
            density,
            log_p_prior,
            node_post_order,
            num_nodes,
            num_samples,
            numba_tree,
            precision,
            round_iters,
            trimmed_post_order,
            w_idx_start,
            weights,
        )

        w_idx_start += round_iters

    sum_of_lls = logsumexp(weights)
    avg_llh = sum_of_lls - np.log(len(weights))

    return avg_llh


def get_num_rounds_and_round_iters_arr(num_iters):
    round_iters = max(floor(num_iters / 10), 10000)
    round_iters = min(round_iters, 100000)
    num_rounds = floor(num_iters / round_iters)
    rem_iters = num_iters - (round_iters * num_rounds)
    if rem_iters > 0:
        num_rounds += 1
        round_iters_arr = np.full(num_rounds, round_iters, dtype=np.int64)
        round_iters_arr[-1] = rem_iters
    else:
        round_iters_arr = np.full(num_rounds, round_iters, dtype=np.int64)
    return num_rounds, round_iters_arr


@njit
def run_batch_of_IS_iters(
    clonal_prevs_for_round,
    density,
    log_p_prior,
    node_post_order,
    num_nodes,
    num_samples,
    numba_tree,
    precision,
    round_iters,
    trimmed_post_order,
    w_idx_start,
    weights,
):
    for j in range(round_iters):
        clonal_prev = clonal_prevs_for_round[j]
        cell_prev = compute_cell_prev_given_clonal_prev(clonal_prev, numba_tree, trimmed_post_order)
        node_llh_arr = compute_node_log_likelihoods(
            cell_prev,
            node_post_order,
            num_nodes,
            num_samples,
            numba_tree,
            density,
            precision,
        )

        loss_ratio = node_llh_arr.sum() + log_p_prior
        weights[j + w_idx_start] = loss_ratio


@njit
def compute_node_log_likelihoods(
    cell_prev,
    node_post_order,
    num_nodes,
    num_samples,
    tree,
    density,
    precision,
):
    node_llh_arr = np.zeros((num_samples, num_nodes))
    for node in node_post_order:
        nb_node = tree.get_node(node)
        snvs = nb_node.snvs

        node_cell_prev = cell_prev[:, node]

        for snv in snvs:
            for sample in range(num_samples):
                numba_sample_dp = nb_node.get_sample_dp(sample, snv)
                mut_ccf = node_cell_prev[sample]
                if density == "binomial":
                    llh = log_pyclone_binomial_pdf(numba_sample_dp, mut_ccf)
                else:
                    llh = log_pyclone_beta_binomial_pdf(numba_sample_dp, mut_ccf, precision)
                node_llh_arr[sample, node] += llh

    return node_llh_arr


@njit
def compute_cell_prev_given_clonal_prev(clonal_prev, tree, trimmed_post_order):
    cell_prev = clonal_prev.copy()
    for node in trimmed_post_order:
        nb_node = tree.get_node(node)
        children = nb_node.children
        child_prevs = cell_prev[:, children].T
        cell_prev[:, node] += child_prevs.sum(axis=0)
    return cell_prev
