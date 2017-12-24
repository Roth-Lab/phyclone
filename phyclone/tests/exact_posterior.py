from collections import defaultdict

import itertools
import numpy as np

from phyclone.consensus import get_clades
from phyclone.math_utils import exp_normalize
from phyclone.tree import MarginalNode, Tree


def get_exact_posterior(data, alpha=1.0):
    grid_size = data[0].grid_size

    log_p = {}

    forests = defaultdict(list)

    for x in get_all_ordered_partitions(data):
        forests[len(x)].append(x)

    for num_nodes in forests:
        for parent_idxs in get_oriented_forests(num_nodes):
            for clusters in forests[num_nodes]:
                t = get_fscrp_tree(alpha, grid_size, clusters, parent_idxs)

                log_p[get_clades(t)] = float(t.log_p_one)

    p, _ = exp_normalize(np.array(list(log_p.values())))

    for clade, p_clade in zip(log_p, p):
        log_p[clade] = p_clade

    return log_p


def get_fscrp_tree(alpha, grid_size, clusters, parent_pointers):
    nodes = []

    for node_idx, node_data in enumerate(clusters):
        node = MarginalNode(node_idx, node_data[0].grid_size)

        for data_point in node_data:
            node.add_data_point(data_point)

        nodes.append(node)

    children = defaultdict(list)

    for node_idx, parent_idx in enumerate(parent_pointers, -1):
        parent_idx -= 1
        if parent_idx >= 0:
            children[parent_idx].append(nodes[node_idx])

    for node_idx, node_children in children.items():
        nodes[node_idx].update_children(node_children)

    tree = Tree.create_tree_from_nodes(1.0, grid_size, nodes, [])

    tree.update_likelihood()

    return tree


def get_oriented_forests(n):
    """
    Implementation of Algorithm O from TAOCP section 7.2.1.6. Generates all canonical n-node oriented forests.

    Written by Jerome Kelleher <jerome.kelleher@well.ox.ac.uk>
    """
    p = [k - 1 for k in range(0, n + 1)]
    k = 1
    while k != 0:
        yield p
        if p[n] > 0:
            p[n] = p[p[n]]
            yield p
        k = n
        while k > 0 and p[k] == 0:
            k -= 1
        if k != 0:
            j = p[k]
            d = k - j
            notDone = True
            while notDone:
                if p[k - d] == p[j]:
                    p[k] = p[j]
                else:
                    p[k] = p[k - d] + d
                if k == n:
                    notDone = False
                else:
                    k += 1


def get_all_partitions(collection):
    collection = list(collection)

    if len(collection) == 1:
        yield [collection]

        return

    first = collection[0]

    for smaller in get_all_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]

        yield [[first]] + smaller


def get_all_ordered_partitions(collection):
    for partition in get_all_partitions(collection):
        for ordered_partition in itertools.permutations(partition):
            yield ordered_partition
