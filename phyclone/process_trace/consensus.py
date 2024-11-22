from collections import defaultdict

import networkx as nx

from phyclone.tree.utils import get_clades


def get_consensus_tree(trees, data=None, threshold=0.5, weighted=False, log_p_list=None):

    clades_counter = clade_probabilities(trees, weighted=weighted, log_p_list=log_p_list)

    consensus_clades = key_above_threshold(clades_counter, threshold)

    consensus_tree = consensus(consensus_clades)

    consensus_tree = relabel(consensus_tree)

    consensus_tree = clean_tree(consensus_tree, data=data)

    return consensus_tree


def consensus(clades):
    """Attempts to build a consensus tree from a set of clades. Returns a DiGraph where nodes are clades."""
    result = nx.DiGraph()

    for clade in clades:
        candiate_supersets = clades.copy()

        parent_clade = find_smallest_superset(candiate_supersets, clade)

        if parent_clade is not None:
            result.add_edge(parent_clade, clade)

        else:
            result.add_node(clade)

    return result


def relabel(graph):
    """Relabels a consensus tree.

    Takes in a DiGraph of clades, return a new DiGraph where nodes are again set of mutation, but with a different
    interpretation. The tranformation used to change the nodes/sets is to start with the original and remove from each
    node the data_points that appear in children clades.
    """
    result = nx.DiGraph()

    for root in roots(graph):
        _relabel(root, result, graph)

    return result


def clean_tree(tree, data=None):
    node_map = {}

    for new_node, old_node in enumerate(nx.dfs_preorder_nodes(tree)):
        node_map[old_node] = new_node

    new_tree = nx.relabel_nodes(tree, node_map)

    idx_map = {}

    for data_points, node in node_map.items():
        idx_map[node] = sorted(data_points)

    nx.set_node_attributes(new_tree, name="idxs", values=idx_map)

    if data is not None:
        name_map = defaultdict(list)

        for node in idx_map:
            for idx in idx_map[node]:
                name_map[node].append(data[idx].name)

        nx.set_node_attributes(new_tree, name="names", values=name_map)

    return new_tree


def clade_probabilities(trees, weighted=False, log_p_list=None):
    """Return a clade probabilities."""
    clades_counter = defaultdict(float)

    for i, tree in enumerate(trees):
        tree_clades = get_clades(tree)
        for clade in tree_clades:
            if weighted:
                clades_counter[clade] += log_p_list[i]

            else:
                clades_counter[clade] += 1

    if not weighted:
        for clade in clades_counter:
            clades_counter[clade] = clades_counter[clade] / len(trees)

    return clades_counter


def key_above_threshold(counter, threshold):
    """Only keeps the keys in a dict above or equal the threshold"""
    return set([key for key, value in counter.items() if value > threshold])


def _relabel(node, transformed, original):
    result = set(node)

    for _, children in original.out_edges(node):
        for mutation in children:
            result.remove(mutation)

    result = frozenset(result)

    transformed.add_node(result)

    for _, children in original.out_edges(node):
        transformed.add_edge(result, _relabel(children, transformed, original))

    return result


def roots(graph):
    return [n for n in graph.nodes() if len(graph.in_edges(n)) == 0]


def find_smallest_superset(set_of_sets, query_set):
    # Remove the query set from set of candidate supersets if present
    set_of_sets.discard(query_set)

    # Intialisation
    smallest_superset_size = float("inf")

    smallest_superset = None

    # Loop through candidate supersets looking for the smallest one
    for candidate_superset in set_of_sets:
        # Skip sets which aren't supersets of query
        if not candidate_superset.issuperset(query_set):
            continue

        candidate_superset_size = len(candidate_superset)

        if candidate_superset_size == smallest_superset_size:
            raise Exception("Inconsistent set of clades")

        if candidate_superset_size < smallest_superset_size:
            smallest_superset_size = candidate_superset_size

            smallest_superset = candidate_superset

    return smallest_superset
