import math
import numpy as np
import random

import phyclone.math_utils

from phyclone.tree import MarginalNode, Tree


class DataPointSwapSampler(object):
    def sample_tree(self, data, tree):
        new_tree = tree.copy()

        labels = new_tree.labels

        idx_1, idx_2 = random.sample(list(labels.keys()), 2)

        if labels[idx_1] == labels[idx_2]:
            return tree

        new_tree._nodes[labels[idx_1]].remove_data_point(data[idx_1])

        new_tree._nodes[labels[idx_2]].remove_data_point(data[idx_2])

        new_tree._nodes[labels[idx_1]].add_data_point(data[idx_2])

        new_tree._nodes[labels[idx_2]].add_data_point(data[idx_1])

        new_tree.update_likelihood()

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


class SimpleSampler(object):
    def sample_tree(self, data, tree):
        new_tree = tree.copy()

        node = random.choice(list(new_tree.nodes.values()))

        if len(node.data) == 1:
            parent = new_tree.get_parent_node(node)

            if parent is None:
                return tree

            new_tree.remove_subtree(new_tree.get_subtree(node))

            new_tree.add_subtree(
                Tree.create_tree_from_nodes(new_tree.alpha, new_tree.grid_size, Tree.get_nodes(node)[1:], [])
            )

            parent.add_data_point(node.data[0])

        else:
            data_point = random.choice(node.data)

            node.remove_data_point(data_point)

            if phyclone.math_utils.bernoulli_rvs():
                parent = new_tree.get_parent_node(node)

                if parent is None:
                    return tree

                parent.add_data_point(data_point)

                assert parent in list(new_tree.nodes.values())

                assert data_point in parent.data

                assert data_point.idx in new_tree.data_points

            else:
                idx = new_tree.new_node_idx

                new_node = MarginalNode(idx, new_tree.grid_size, [])

                new_node.add_data_point(data_point)

                node.add_child_node(new_node)

                new_tree._nodes[new_node.idx] = new_node

                new_tree._graph.add_edge(node.idx, new_node.idx)

            assert data_point.idx in new_tree.data_points

        new_tree.update_likelihood()

        u = random.random()

#         print(new_tree.log_p_one, tree.log_p_one)

        if new_tree.log_p_one - tree.log_p_one > np.log(u):
            tree = new_tree

        return tree


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

        parent_idx = random.choice([x.idx for x in remaining_nodes] + [None, ])

        new_tree.add_subtree(subtree, parent_idx)

        old_log_p = tree.log_p_one

        new_log_p = new_tree.log_p_one

        u = random.random()

        if new_log_p - old_log_p > math.log(u):
            tree = new_tree

        return tree
