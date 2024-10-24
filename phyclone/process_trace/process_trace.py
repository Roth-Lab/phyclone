import gzip
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from phyclone.process_trace.consensus import get_consensus_tree
from phyclone.process_trace.map import get_map_node_ccfs
from phyclone.process_trace.utils import print_string_to_file
from phyclone.tree import Tree
from phyclone.utils.math import exp_normalize


def write_map_results(
    in_file,
    out_table_file,
    out_tree_file,
    out_log_probs_file=None,
    map_type="frequency",
):

    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    data = results[0]["data"]

    chain_num = 0

    if map_type == "frequency":

        topologies = create_topology_dict_from_trace(results)

        df = create_topology_dataframe(topologies.values())
        df = df.sort_values(by="count", ascending=False)

        map_iter = df["iter"].iloc[0]
        chain_num = df["chain_num"].iloc[0]
    else:
        map_iter = 0

        map_val = float("-inf")

        for curr_chain_num, chain_results in results.items():
            for i, x in enumerate(chain_results["trace"]):
                if x["log_p_one"] > map_val:
                    map_iter = i

                    map_val = x["log_p_one"]
                    chain_num = curr_chain_num

    tree = Tree.from_dict(results[chain_num]["trace"][map_iter]["tree"])

    clusters = results[0].get("clusters", None)

    table = get_clone_table(data, results[0]["samples"], tree, clusters=clusters)

    _create_results_output_files(
        out_log_probs_file, out_table_file, out_tree_file, results, table, tree
    )


def create_topology_dict_from_trace(trace):
    topologies = dict()
    for chain_num, chain_result in trace.items():
        chain_trace = chain_result["trace"]
        for i, x in enumerate(chain_trace):
            curr_tree = Tree.from_dict(x["tree"])
            count_topology(topologies, x, i, curr_tree, chain_num)
    return topologies


def write_topology_report(in_file, out_file):

    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    topologies = create_topology_dict_from_trace(results)

    df = create_topology_dataframe(topologies.values())
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(out_file, index=False, sep="\t")


def count_topology(topologies, x, i, x_top, chain_num=0):
    if x_top in topologies:
        topology = topologies[x_top]
        topology["count"] += 1
        curr_log_p_one = x["log_p_one"]
        if curr_log_p_one > topology["log_p_joint_max"]:
            topology["log_p_joint_max"] = curr_log_p_one
            topology["iter"] = i
            topology["topology"] = x_top
            topology["chain_num"] = chain_num
    else:
        log_mult = x_top.multiplicity
        topologies[x_top] = {
            "topology": x_top,
            "count": 1,
            "log_p_joint_max": x["log_p_one"],
            "iter": i,
            "chain_num": chain_num,
            "multiplicity": np.exp(log_mult),
            "log_multiplicity": log_mult,
        }


def _create_results_output_files(
    out_log_probs_file, out_table_file, out_tree_file, results, table, tree
):
    table.to_csv(out_table_file, index=False, sep="\t")
    print_string_to_file(tree.to_newick_string(), out_tree_file)
    if out_log_probs_file:
        log_probs_table = pd.DataFrame(
            results["trace"], columns=["iter", "time", "log_p_one"]
        )
        log_probs_table.to_csv(out_log_probs_file, index=False, sep="\t")


def create_topology_dataframe(topologies):
    for topology in topologies:
        tree = topology["topology"]
        topology["topology"] = tree.to_newick_string()

    df = pd.DataFrame(topologies)
    return df


def write_consensus_results(
    in_file,
    out_table_file,
    out_tree_file,
    out_log_probs_file=None,
    consensus_threshold=0.5,
    weight_type="counts",
):

    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    data = results[0]["data"]

    trees = []
    probs = []

    if weight_type == "counts":
        weighted_consensus = False
        for chain_results in results.values():
            trees.extend([Tree.from_dict(x["tree"]) for x in chain_results["trace"]])
    else:
        weighted_consensus = True
        for chain_results in results.values():
            trees.extend([Tree.from_dict(x["tree"]) for x in chain_results["trace"]])
            probs.extend([x["log_p_one"] for x in chain_results["trace"]])

    if weighted_consensus:
        probs = np.array(probs)
        probs, norm = exp_normalize(probs)

    graph = get_consensus_tree(
        trees,
        data=data,
        threshold=consensus_threshold,
        weighted=weighted_consensus,
        log_p_list=probs,
    )

    tree = get_tree_from_consensus_graph(data, graph)

    clusters = results[0].get("clusters", None)

    table = get_clone_table(data, results[0]["samples"], tree, clusters=clusters)

    table = pd.DataFrame(table)

    _create_results_output_files(
        out_log_probs_file, out_table_file, out_tree_file, results, table, tree
    )


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

    tree = from_dict_nx(data, {"graph": nx.to_dict_of_dicts(graph), "labels": labels})

    tree.update()

    return tree


def from_dict_nx(data, tree_dict):
    new = Tree(data[0].grid_size)

    data = dict(zip([x.idx for x in data], data))

    for node in tree_dict["graph"].keys():
        if node == "root":
            continue
        new._add_node(node)

    for parent, children in tree_dict["graph"].items():
        parent_idx = new._node_indices[parent]
        for child in children.keys():
            child_idx = new._node_indices[child]
            new._graph.add_edge(parent_idx, child_idx, None)

    for idx, node in tree_dict["labels"].items():
        new._internal_add_data_point_to_node(True, data[idx], node)

    new.update()

    return new


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


def create_main_run_output(cluster_file, out_file, results):
    for chain_result in results.values():
        if cluster_file is not None:
            chain_result["clusters"] = pd.read_csv(cluster_file, sep="\t")[
                ["mutation_id", "cluster_id"]
            ].drop_duplicates()
    with gzip.GzipFile(out_file, mode="wb") as fh:
        pickle.dump(results, fh)
