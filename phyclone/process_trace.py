import gzip
import os
import pickle
from io import StringIO

import Bio.Phylo
import networkx as nx
import numpy as np
import pandas as pd
from numba import set_num_threads

from phyclone.consensus import get_consensus_tree
from phyclone.map import get_map_node_ccfs
from phyclone.math_utils import exp_normalize
from phyclone.smc.kernels.fully_adapted import _get_cached_proposal_dist
from phyclone.tree import Tree
from phyclone.tree_utils import (
    compute_log_S,
    _cache_ratio,
    _convolve_two_children,
)


def write_map_results(in_file, out_table_file, out_tree_file, out_log_probs_file=None):
    set_num_threads(1)
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    # prior_type = results['run_info']['prior']
    data = results["data"]

    # if prior_type == 'uniform':
    #     topologies = []
    #
    #     for i, x in enumerate(results["trace"]):
    #         curr_tree = Tree.from_dict(data, x["tree"])
    #         count_topology(topologies, x, i, curr_tree)
    #
    #     df = _create_topology_dataframe(topologies)
    #     df = df.sort_values(by="count", ascending=False)
    #
    #     map_iter = df['iter'].iloc[0]
    # else:
    #     map_iter = 0
    #
    #     map_val = float("-inf")
    #
    #     for i, x in enumerate(results["trace"]):
    #         if x["log_p_one"] > map_val:
    #             map_iter = i
    #
    #             map_val = x["log_p_one"]
    map_iter = 0

    map_val = float("-inf")

    for i, x in enumerate(results["trace"]):
        if x["log_p_one"] > map_val:
            map_iter = i

            map_val = x["log_p_one"]

    tree = Tree.from_dict(data, results["trace"][map_iter]["tree"])

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    _create_results_output_files(
        out_log_probs_file, out_table_file, out_tree_file, results, table, tree
    )


def write_topology_report(in_file, out_file):
    set_num_threads(1)

    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    topologies = []

    data = results["data"]

    data_arr = np.array(list(data))

    data_index_dict = dict(
        zip(np.arange(len(data_arr)), [data_point.idx for data_point in data_arr])
    )

    parent_child_arr = np.zeros((len(data_arr), len(data_arr)))

    for i, x in enumerate(results["trace"]):
        curr_tree = Tree.from_dict(data, x["tree"])
        count_parent_child_relationships(curr_tree, data_index_dict, parent_child_arr)
        count_topology(topologies, x, i, curr_tree)

    df = _create_topology_dataframe(topologies)
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(out_file, index=False, sep="\t")

    _create_parent_child_out_files(data_arr, out_file, parent_child_arr, results)


def _create_parent_child_out_files(data_arr, out_file, parent_child_arr, results):
    parent_child_df = _create_parent_child_matrix_df(data_arr, parent_child_arr)
    parent_child_counts_out = os.path.join(
        os.path.dirname(out_file), "parent_child_matrix_counts.tsv"
    )
    parent_child_df.to_csv(parent_child_counts_out, index=False, sep="\t")
    parent_child_out = os.path.join(
        os.path.dirname(out_file), "parent_child_matrix.tsv"
    )
    trace_len = len(results["trace"])
    parent_child_probs_arr = parent_child_arr / trace_len
    parent_child_probs_df = _create_parent_child_matrix_df(
        data_arr, parent_child_probs_arr
    )
    parent_child_probs_df.to_csv(parent_child_out, index=False, sep="\t")


def _create_parent_child_matrix_df(data_arr, parent_child_arr):
    parent_child_df = pd.DataFrame(
        parent_child_arr,
        columns=[str(data_point.name) for data_point in data_arr],
        index=[str(data_point.name) for data_point in data_arr],
    )
    parent_child_df = parent_child_df.reset_index()
    parent_child_df = parent_child_df.rename(columns={"index": "ID"})
    return parent_child_df


def count_parent_child_relationships(curr_tree, data_index_dict, parent_child_arr):
    curr_graph = curr_tree.graph
    curr_node_data = curr_tree.node_data
    for node in nx.dfs_preorder_nodes(curr_graph):
        dp_in_node = curr_node_data[node]
        children = list(curr_graph.successors(node))
        for dp in dp_in_node:
            for child in children:
                dp_in_child_node = curr_node_data[child]
                for child_dp in dp_in_child_node:
                    parent_child_arr[
                        data_index_dict[dp.idx], data_index_dict[child_dp.idx]
                    ] += 1


def count_topology(topologies, x, i, x_top):
    found = False
    for topology in topologies:
        top = topology["topology"]
        if top == x_top:
            topology["count"] += 1
            curr_log_p = x["log_p_one"]
            if curr_log_p > topology["log_p_max"]:
                topology["log_p_max"] = curr_log_p
                topology["iter"] = i
            found = True
            break
    if not found:
        topologies.append(
            {
                "topology": x_top,
                "count": 1,
                "log_p_max": x["log_p_one"],
                "iter": i,
                "multiplicity": np.exp(x_top.multiplicity),
            }
        )


def _create_results_output_files(
    out_log_probs_file, out_table_file, out_tree_file, results, table, tree
):
    table.to_csv(out_table_file, index=False, sep="\t")
    Bio.Phylo.write(
        get_bp_tree_from_graph(tree.graph), out_tree_file, "newick", plain=True
    )
    if out_log_probs_file:
        log_probs_table = pd.DataFrame(
            results["trace"], columns=["iter", "time", "log_p"]
        )
        log_probs_table.to_csv(out_log_probs_file, index=False, sep="\t")


def _create_topology_dataframe(topologies):
    for topology in topologies:
        tmp_str_io = StringIO()
        tree = topology["topology"]
        Bio.Phylo.write(
            get_bp_tree_from_graph(tree.graph), tmp_str_io, "newick", plain=True
        )
        as_str = tmp_str_io.getvalue().rstrip()
        topology["topology"] = as_str

    df = pd.DataFrame(topologies)
    df["multiplicity_corrected_count"] = df["count"] // df["multiplicity"]
    return df


def write_consensus_results(
    in_file,
    out_table_file,
    out_tree_file,
    out_log_probs_file=None,
    consensus_threshold=0.5,
    weighted_consensus=True,
):
    set_num_threads(1)
    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    data = results["data"]

    trees = [Tree.from_dict(data, x["tree"]) for x in results["trace"]]

    probs = np.array([x["log_p_one"] for x in results["trace"]])

    probs, norm = exp_normalize(probs)

    graph = get_consensus_tree(
        trees,
        data=data,
        threshold=consensus_threshold,
        weighted=weighted_consensus,
        log_p_list=probs,
    )

    tree = get_tree_from_consensus_graph(data, graph)

    clusters = results.get("clusters", None)

    table = get_clone_table(data, results["samples"], tree, clusters=clusters)

    table = pd.DataFrame(table)

    _create_results_output_files(
        out_log_probs_file, out_table_file, out_tree_file, results, table, tree
    )


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
            df.append(
                {
                    "mutation_id": data[idx].name,
                    "clone_id": tree.labels[idx],
                }
            )

            clone_muts.add(data[idx].name)

        for x in data:
            if x.name not in clone_muts:
                df.append({"mutation_id": x.name, "clone_id": -1})

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id"])

        df = df.sort_values(by=["clone_id", "mutation_id"])

    else:
        for idx in tree.labels:
            muts = clusters[clusters["cluster_id"] == int(data[idx].name)][
                "mutation_id"
            ]

            for mut in muts:
                df.append(
                    {
                        "mutation_id": mut,
                        "clone_id": tree.labels[idx],
                        "cluster_id": int(data[idx].name),
                    }
                )

                clone_muts.add(mut)

        clusters = clusters.set_index("mutation_id")

        for mut in clusters.index.values:
            if mut not in clone_muts:
                df.append(
                    {
                        "mutation_id": mut,
                        "clone_id": -1,
                        "cluster_id": clusters.loc[mut].values[0],
                    }
                )

        df = pd.DataFrame(df, columns=["mutation_id", "clone_id", "cluster_id"])

        df = df.sort_values(by=["clone_id", "cluster_id", "mutation_id"])

    return df


def _create_main_run_output(cluster_file, out_file, results):
    if cluster_file is not None:
        results["clusters"] = pd.read_csv(cluster_file, sep="\t")[
            ["mutation_id", "cluster_id"]
        ].drop_duplicates()
    with gzip.GzipFile(out_file, mode="wb") as fh:
        pickle.dump(results, fh)

    cache_txt_file = os.path.join(os.path.dirname(out_file), 'cache_info.txt')
    create_cache_info_file(cache_txt_file)


def create_cache_info_file(out_file):
    with open(out_file, "w") as f:
        print(
            "compute_s cache info: {}, hit ratio: {}".format(
                compute_log_S.cache_info(), _cache_ratio(compute_log_S.cache_info())
            ),
            file=f,
        )
        print(
            "_convolve_two_children cache info: {}, hit ratio: {}".format(
                _convolve_two_children.cache_info(),
                _cache_ratio(_convolve_two_children.cache_info()),
            ),
            file=f,
        )
        print(
            "_get_cached_proposal_dist cache info: {}, hit ratio: {}".format(
                _get_cached_proposal_dist.cache_info(),
                _cache_ratio(_get_cached_proposal_dist.cache_info()),
            ),
            file=f,
        )
