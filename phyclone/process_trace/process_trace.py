import gzip
import pickle
from sys import maxsize

import networkx as nx
import numpy as np
import pandas as pd

from phyclone.process_trace.consensus import get_consensus_tree
from phyclone.process_trace.map import get_map_node_ccfs_and_clonal_prev_dicts
from phyclone.process_trace.utils import print_string_to_file
from phyclone.tree import Tree
import tarfile
import os
import tempfile
from scipy.special import softmax


def write_map_results(
    in_file,
    out_table_file,
    out_tree_file,
    map_type="joint-likelihood",
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

    _create_results_output_files(out_table_file, out_tree_file, table, tree)


def create_topology_dict_from_trace(trace):
    topologies = dict()
    for chain_num, chain_result in trace.items():
        chain_trace = chain_result["trace"]
        for i, x in enumerate(chain_trace):
            curr_tree = Tree.from_dict(x["tree"])
            count_topology(topologies, x, i, curr_tree, chain_num)
    return topologies


def write_topology_report(in_file, out_file, topologies_archive=None, top_trees=float('inf')):
    if top_trees == maxsize:
        top_trees = float('inf')
    print()
    print("#" * 100)
    print("PhyClone - Topology Report")
    print("#" * 100)

    with gzip.GzipFile(in_file, "rb") as fh:
        results = pickle.load(fh)

    print("\nExtracting unique topologies from sample trace.")
    topologies_dict = create_topology_dict_from_trace(results)

    topology_df = create_topology_dataframe(topologies_dict.values())
    topology_df.to_csv(out_file, index=False, sep="\t")

    print("Topology report created, saved as: {}".format(out_file))

    if topologies_archive is not None:
        print()
        print("#" * 50)
        if top_trees == float('inf'):
            top_trees_statement = "for all {}".format(len(topologies_dict))
        else:
            top_trees_statement = "for the top {}".format(top_trees)
            if top_trees > len(topologies_dict):
                print("Warning: Number of top trees requested ({}) "
                      "is greater than the total number of "
                      "uniquely sampled topologies ({}).".format(top_trees, len(topologies_dict)))
                top_trees_statement = "for all {}".format(len(topologies_dict))
        print("\nBuilding PhyClone topologies archive {} uniquely sampled topologies.".format(top_trees_statement))
        create_topologies_archive(topology_df, results, top_trees, topologies_dict, topologies_archive)
        print("Topologies archive created, saved as: {}".format(topologies_archive))
    print('\nFinished.')
    print("#" * 100)


def create_topologies_archive(topology_df, results, top_trees, topologies_dict, topologies_archive):
    filename_template = '{}_results_table.tsv'
    nwk_template = '{}.nwk'
    clusters = results[0].get("clusters", None)
    data = results[0]["data"]
    samples = results[0]["samples"]
    with tarfile.open(topologies_archive, "w:gz") as archive:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for tree, values in topologies_dict.items():
                row = topology_df.loc[(topology_df["topology"] == values['topology'])
                                      & (topology_df["count"] == values['count'])
                                      & (topology_df["log_p_joint_max"] == values['log_p_joint_max'])
                                      & (topology_df["iter"] == values['iter'])
                                      & (topology_df["chain_num"] == values['chain_num'])]
                assert len(row) == 1
                topology_id = row['topology_id'].values[0]
                topology_rank = int(topology_id[2:])
                if topology_rank >= top_trees:
                    continue
                table = get_clone_table(data, samples, tree, clusters=clusters)
                filename = filename_template.format(topology_id)
                filepath = os.path.join(tmp_dir, filename)
                table.to_csv(filepath, index=False, sep="\t")
                archive.add(filepath, arcname=str(os.path.join(topology_id, filename)))
                nwk_filename = nwk_template.format(topology_id)
                nwk_path = os.path.join(tmp_dir, nwk_filename)
                print_string_to_file(tree.to_newick_string(), nwk_path)
                archive.add(nwk_path, arcname=str(os.path.join(topology_id, nwk_filename)))


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


def _create_results_output_files(out_table_file, out_tree_file, table, tree):
    table.to_csv(out_table_file, index=False, sep="\t")
    print_string_to_file(tree.to_newick_string(), out_tree_file)


def create_topology_dataframe(topologies):
    for topology in topologies:
        tree = topology["topology"]
        topology["topology"] = tree.to_newick_string()

    df = pd.DataFrame(topologies)
    df = df.sort_values(by="log_p_joint_max", ascending=False, ignore_index=True)
    df.insert(0, 'topology_id', 't_' + df.index.astype(str))
    return df


def write_consensus_results(
    in_file,
    out_table_file,
    out_tree_file,
    consensus_threshold=0.5,
    weight_type="joint-likelihood",
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
        topologies = create_topology_dict_from_trace(results)

        for tree, top_info in topologies.items():
            trees.append(tree)
            probs.append(top_info["log_p_joint_max"] + np.log(top_info["count"]))

    if weighted_consensus:
        probs = np.array(probs)
        probs = softmax(probs)

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

    _create_results_output_files(out_table_file, out_tree_file, table, tree)


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

    ccfs, clonal_prev_dict = get_map_node_ccfs_and_clonal_prev_dicts(tree)

    samples_idx_dict = {k: v for v, k in enumerate(samples)}

    labels["sample_id"] = [samples]*len(labels)

    labels = labels.explode("sample_id")
    grouped = labels.groupby(["clone_id", 'sample_id'])

    df_list = []

    for name, group in grouped:
        clone_id, sample_id = name

        if clone_id in ccfs:
            group["ccf"] = ccfs[clone_id][samples_idx_dict[sample_id]]
            group["clonal_prev"] = clonal_prev_dict[clone_id][samples_idx_dict[sample_id]]
        else:
            group["ccf"] = -1
            group["clonal_prev"] = -1

        df_list.append(group)

    return pd.concat(df_list, ignore_index=True)


def get_labels_table(data, tree, clusters=None):
    df_records_list = []

    clone_muts = set()

    if clusters is None:
        tree_labels = tree.labels
        for idx in tree_labels:
            df_records_list.append(
                {
                    "mutation_id": data[idx].name,
                    "clone_id": tree_labels[idx],
                }
            )

            clone_muts.add(data[idx].name)

        for x in data:
            if x.name not in clone_muts:
                df_records_list.append({"mutation_id": x.name, "clone_id": -1})

        df = pd.DataFrame(df_records_list)

        df = df.sort_values(by=["clone_id", "mutation_id"])

    else:
        tree_labels = tree.labels

        clusters_grouped = clusters.groupby("cluster_id")
        for idx in tree_labels:

            cluster_id = int(data[idx].name)
            clone_id = tree_labels[idx]

            muts = clusters_grouped.get_group(cluster_id)["mutation_id"]

            muts_set = muts.unique()

            curr_muts_records = [{"mutation_id": mut, "clone_id": clone_id, "cluster_id": cluster_id}
                                 for mut in muts_set]

            clone_muts.update(muts_set)

            df_records_list.extend(curr_muts_records)

        missing_muts_df = clusters.loc[~clusters['mutation_id'].isin(clone_muts)]

        missing_muts_df = missing_muts_df.copy()

        missing_muts_df["clone_id"] = -1

        df_records_list.extend(missing_muts_df.to_dict("records"))

        df = pd.DataFrame(df_records_list)

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
