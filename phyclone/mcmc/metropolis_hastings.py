import math
import numpy as np
import random

import phyclone.math_utils


class DataPointSwapSampler(object):
    def sample_tree(self, tree):
        data = tree.data

        new_tree = tree.copy()

        labels = new_tree.labels

        idx_1, idx_2 = random.sample(list(labels.keys()), 2)

        node_1 = labels[idx_1]

        node_2 = labels[idx_2]

        if node_1 == node_2:
            return tree

        data_point_1 = data[idx_1]

        assert data_point_1.idx == idx_1

        data_point_2 = data[idx_2]

        assert data_point_2.idx == idx_2

        new_tree.remove_data_point_from_node(data_point_1, node_1)

        new_tree.remove_data_point_from_node(data_point_2, node_2)

        new_tree.add_data_point_to_node(data_point_1, node_2)

        new_tree.add_data_point_to_node(data_point_2, node_1)

        u = random.random()

        if new_tree.log_p_one - tree.log_p_one > np.log(u):
            tree = new_tree

        return tree


class ParentChildSwap(object):
    def sample_tree(self, data, tree):
        if len(tree.nodes) == 1:
            return tree

        new_tree = tree.copy()

        node_1 = random.choice(list(new_tree.nodes.values()))

        node_2 = new_tree.get_parent_node(node_1)

        if node_2 is None:
            return tree

        node_2_parent = new_tree.get_parent_node(node_2)

        if node_2_parent is None:
            return tree

        new_node_1_children = node_2.children

        new_node_2_children = node_1.children

        new_node_1_children.remove(node_1)

        new_node_1_children.append(node_2)

        node_2.update_children(new_node_2_children)

        node_1.update_children(new_node_1_children)

        new_tree._graph.remove_edge(node_2_parent.idx, node_2.idx)

        new_tree._graph.add_edge(node_2_parent.idx, node_1.idx)

        node_2_parent.remove_child_node(node_2)

        node_2_parent.add_child_node(node_1)

        node_2_parent.update()

        new_tree._graph.remove_edge(node_2.idx, node_1.idx)

        new_tree._graph.add_edge(node_1.idx, node_2.idx)

        new_tree.update_likelihood()

        u = random.random()

        if new_tree.log_p_one - tree.log_p_one > np.log(u):
            print('Accepted')
            tree = new_tree

        return tree
#
#
# class SimpleSampler(object):
#     def sample_tree(self, node_data, tree):
#         new_tree = tree.copy()
#
#         node = random.choice(list(new_tree.nodes.values()))
#
#         if len(node.node_data) == 1:
#             parent = new_tree.get_parent_node(node)
#
#             if parent is None:
#                 return tree
#
#             new_tree.remove_subtree(new_tree.get_subtree(node))
#
#             new_tree.add_subtree(
#                 Tree.create_tree_from_nodes(new_tree.alpha, new_tree.grid_size, Tree.get_nodes(node)[1:], [])
#             )
#
#             parent.add_data_point(node.node_data[0])
#
#         else:
#             data_point = random.choice(node.node_data)
#
#             node.remove_data_point(data_point)
#
#             if phyclone.math_utils.bernoulli_rvs():
#                 new_node = random.choice(list(new_tree.nodes.values()))
#
#                 new_node.add_data_point(data_point)
#
#                 assert data_point in new_node.node_data
#
#                 assert data_point.idx in new_tree.data_points
#
#             else:
#                 idx = new_tree.new_node_idx
#
#                 new_node = MarginalNode(idx, new_tree.grid_size, [])
#
#                 new_node.add_data_point(data_point)
#
#                 node.add_child_node(new_node)
#
#                 new_tree._nodes[new_node.idx] = new_node
#
#                 new_tree._graph.add_edge(node.idx, new_node.idx)
#
#             assert data_point.idx in new_tree.data_points
#
#         new_tree.update_likelihood()
#
#         u = random.random()
#
# #         print(new_tree.log_p_one, tree.log_p_one)
#
#         if new_tree.log_p_one - tree.log_p_one > np.log(u):
#             tree = new_tree
#
#         return tree


class OutlierSampler(object):
    def sample_tree(self, tree):
        outliers = list(tree.outliers)

        random.shuffle(outliers)

        for data_point in outliers:
            log_p = {-1: tree.log_p}

            tree.remove_data_point_from_outliers(data_point)

            for node in tree.nodes:
                tree.add_data_point_to_node(data_point, node)

                log_p[node] = tree.log_p

                tree.remove_data_point_from_node(data_point, node)

            p, _ = phyclone.math_utils.exp_normalize(np.array(list(log_p.values())).astype(float))

            x = phyclone.math_utils.discrete_rvs(p)

            node = list(log_p.keys())[x]

            if node == -1:
                tree.add_data_point_to_outliers(data_point)

            else:
                tree.add_data_point_to_node(data_point, node)

        return tree


class PruneRegraphSampler(object):
    def sample_tree(self, tree):
        if len(tree.nodes) <= 1:
            return tree

        new_tree = tree.copy()

        subtree_root = random.choice(new_tree.nodes)

        subtree = new_tree.get_subtree(subtree_root)

        new_tree.remove_subtree(subtree)

        remaining_nodes = new_tree.nodes

        if len(remaining_nodes) == 0:
            return tree

        parent = random.choice(remaining_nodes + [None, ])

        new_tree.add_subtree(subtree, parent=parent)

        old_log_p = tree.log_p_one

        new_log_p = new_tree.log_p_one

        u = random.random()

        if new_log_p - old_log_p > math.log(u):
            tree = new_tree

        return tree
