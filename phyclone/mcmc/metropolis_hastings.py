from __future__ import division

import math
import random


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
