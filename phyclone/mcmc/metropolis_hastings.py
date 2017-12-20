import math
import numpy as np
import random

import phyclone.math_utils


class OutlierSampler(object):
    def sample_tree(self, tree):
        outliers = list(tree.outliers)

        random.shuffle(outliers)

        for data_point in outliers:
            log_p = {-1: tree.log_p}

            tree.outliers.remove(data_point)

            for node in tree.nodes.values():
                node.add_data_point(data_point)

                tree._update_ancestor_nodes(node)

                log_p[node.idx] = tree.log_p

                node.remove_data_point(data_point)

                tree._update_ancestor_nodes(node)

            p, _ = phyclone.math_utils.exp_normalize(np.array(list(log_p.values())).astype(float))

            x = phyclone.math_utils.discrete_rvs(p)

            node_idx = list(log_p.keys())[x]

            if node_idx == -1:
                tree.outliers.append(data_point)

            else:
                tree.nodes[node_idx].add_data_point(data_point)

                tree._update_ancestor_nodes(tree.nodes[node_idx])

        return tree


class PruneRegraphSampler(object):
    def sample_tree(self, data, tree):
        if len(tree.nodes) <= 1:
            return tree

        new_tree = tree.copy()

        nodes = list(new_tree.nodes.values())

        subtree_root = random.choice(nodes)

        subtree = new_tree.get_subtree(subtree_root)

        new_tree.remove_subtree(subtree)

        remaining_nodes = list(new_tree.nodes.values())

        if len(remaining_nodes) == 0:
            return tree

        parent = random.choice(remaining_nodes)

        new_tree.add_subtree(subtree, parent)

        old_log_p = tree.log_p_one

        new_log_p = new_tree.log_p_one

        u = random.random()

        if new_log_p - old_log_p > math.log(u):
            tree = new_tree

        return tree
