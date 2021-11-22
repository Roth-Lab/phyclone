from collections import defaultdict

import itertools
import numpy as np

from phyclone.consensus import get_clades
from phyclone.math_utils import exp_normalize
from phyclone.tree import Tree


def get_exact_posterior(data, tree_dist, alpha=1.0):
    grid_size = data[0].grid_size

    log_p = {}

    forests = defaultdict(list)

    for x in get_all_ordered_partitions(data):
        forests[len(x)].append(x)

    for num_nodes in forests:
        for parent_idxs in get_oriented_forests(num_nodes):
            for clusters in forests[num_nodes]:
                t = get_fscrp_tree(alpha, grid_size, clusters, parent_idxs)

                log_p[get_clades(t)] = float(tree_dist.log_p_one(t))

    p, _ = exp_normalize(np.array(list(log_p.values())))

    for clade, p_clade in zip(log_p, p):
        log_p[clade] = p_clade

    return log_p


def get_fscrp_tree(alpha, grid_size, clusters, parent_pointers):
    data = defaultdict(list)

    for node, node_data in enumerate(clusters):
        for data_point in node_data:
            data[node].append(data_point)

    tree = Tree(grid_size)

    for child, parent in enumerate(parent_pointers, -1):
        parent -= 1

        if parent == -1:
            tree._graph.add_edge('root', child)

        elif parent >= 0:
            tree._graph.add_edge(parent, child)

        if child >= 0:
            tree._add_node(child)

            for data_point in data[child]:
                tree.add_data_point_to_node(data_point, child)

    tree.update()

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
