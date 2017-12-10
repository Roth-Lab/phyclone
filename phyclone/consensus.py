from __future__ import division

from collections import defaultdict

import networkx as nx


def get_consensus_tree(trees, threshold=0.5, weighted=False):
    clades_counter = clade_probabilities(trees, weighted=weighted)

    consensus_clades = key_above_threshold(clades_counter, threshold)

    consensus_tree = consensus(consensus_clades)

    consensus_tree = relabel(consensus_tree)

    consensus_tree = clean_tree(consensus_tree)

    return consensus_tree


def consensus(clades):
    """ Attempts to build a consensus tree from a set of clades. Returns a DiGraph where nodes are clades.
    """
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
    """ Relabels a consensus tree.

    Takes in a DiGraph of clades, return a new DiGraph where nodes are again set of mutation, but with a different
    interpretation. The tranformation used to change the nodes/sets is to start with the original and remove from each
    node the data_points that appear in children clades.
    """
    result = nx.DiGraph()

    for root in roots(graph):
        _relabel(root, result, graph)

    return result


def clean_tree(tree):
    new_tree = nx.DiGraph()

    node_ids = {}

    for i, node in enumerate(nx.dfs_preorder_nodes(tree)):
        node_ids[node] = "Node {0}".format(i + 1)

        new_tree.add_node(node_ids[node], data_points=sorted(node))

    for node in node_ids:
        for child in tree.successors(node):
            new_tree.add_edge(node_ids[node], node_ids[child])

    return new_tree


def clade_probabilities(trees, weighted=False):
    """ Return a clade probabilities.
    """
    clades_counter = defaultdict(float)

    for tree in trees:
        for clade in clades(tree):
            if weighted:
                clades_counter[clade] += trees[tree]

            else:
                clades_counter[clade] += 1

    if not weighted:
        for clade in clades_counter:
            clades_counter[clade] = clades_counter[clade] / len(trees)

    return clades_counter


def key_above_threshold(counter, threshold):
    """ Only keeps the keys in a dict above or equal the threshold
    """
    return set([key for key, value in counter.iteritems() if value > threshold])


def clades(tree):
    result = set()

    for root in tree.roots:
        _clades(result, root, tree)

    return result


def _clades(clades, node, tree):
    current_clade = set()

    for mutation in tree.nodes[node.idx].data:
        current_clade.add(mutation.idx)

    for child in node.children:
        for mutation in _clades(clades, child, tree):
            current_clade.add(mutation)

    clades.add(frozenset(current_clade))

    return current_clade


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
    smallest_superset_size = float('inf')

    smallest_superset = None

    # Loop through candidate supersets looking for the smallest one
    for candidate_superset in set_of_sets:
        # Skip sets which aren't supersets of query
        if not candidate_superset.issuperset(query_set):
            continue

        candidate_superset_size = len(candidate_superset)

        if candidate_superset_size == smallest_superset_size:
            raise Exception('Inconsistent set of clades')

        if candidate_superset_size < smallest_superset_size:
            smallest_superset_size = candidate_superset_size

            smallest_superset = candidate_superset

    return smallest_superset
