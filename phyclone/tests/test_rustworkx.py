import numpy as np

from phyclone.tests.old_implementations import OldTree
from phyclone.tree.utils import get_clades

import unittest
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.tests.simulate import simulate_binomial_data


class Test(unittest.TestCase):

    def setUp(self):
        self.outlier_prob = 0.0

        self._rng = np.random.default_rng(12345)

        self.tree_dist = FSCRPDistribution(1.0)

        self.tree_joint_dist = TreeJointDistribution(self.tree_dist)

    def build_cherry_tree(self, tree_class, data=None):
        if data is None:
            data = self.build_six_datapoints()

        tree = tree_class.get_single_node_tree(data[:2])

        exp_n_1 = tree.create_root_node(children=[], data=data[2:4])

        expected_tree_roots = tree.roots

        exp_n_2 = tree.create_root_node(children=expected_tree_roots, data=data[4:])

        return tree

    def build_linear_tree(self, tree_class, data=None):
        if data is None:
            data = self.build_six_datapoints()

        tree = tree_class.get_single_node_tree(data[:2])

        expected_tree_roots = tree.roots

        exp_n_1 = tree.create_root_node(children=expected_tree_roots, data=data[2:4])

        expected_tree_roots = tree.roots

        exp_n_2 = tree.create_root_node(children=[exp_n_1], data=data[4:])

        return tree

    def build_six_datapoints(self):
        n = 100
        p = 1.0
        data = self._create_data_points(6, n, p)
        return data

    def test_copy(self):
        original_tree = self.build_cherry_tree(Tree)

        copied_tree = original_tree.copy()

        original_nodes = original_tree._graph.nodes()
        copied_nodes = copied_tree._graph.nodes()

        for orig_node, copied_node in zip(original_nodes, copied_nodes):
            self.assertIsNot(orig_node, copied_node)
            np.testing.assert_array_equal(copied_node.log_r, orig_node.log_r)
            np.testing.assert_array_equal(copied_node.log_p, orig_node.log_p)
            self.assertIsNot(copied_node.log_r, orig_node.log_r)
            self.assertIsNot(copied_node.log_p, orig_node.log_p)
            self.assertEqual(copied_node.node_id, orig_node.node_id)

            copied_node.log_r += 1

            self.assertFalse(np.array_equal(copied_node.log_r, orig_node.log_r))

        self.assertEqual(copied_tree, original_tree)
        self.assertIsNot(copied_tree, original_tree)

    # def test_cherry_tree_to_dict_representation_v2(self):
    #
    #     data = self.build_six_datapoints()
    #
    #     expected_tree = self.build_cherry_tree(Tree, data)
    #
    #     actual_tree = self.build_cherry_tree(Tree, data)
    #
    #     actual_dict = actual_tree.to_dict()
    #
    #     expected_dict = expected_tree.to_dict_old()
    #
    #     expected_tree2 = Tree.from_dict_old(data, expected_dict)
    #
    #     actual_tree2 = Tree.from_dict(data, actual_dict)
    #
    #     self.assertEqual(actual_tree2, expected_tree)
    #     self.assertEqual(actual_tree2, actual_tree)
    #     self.assertEqual(actual_tree2, expected_tree2)

    def test_single_node_tree_to_dict_representation(self):
        n = 100
        p = 1.0

        data = self._create_data_points(4, n, p)

        actual_tree = Tree.get_single_node_tree(data)

        expected_tree = OldTree.get_single_node_tree(data)

        actual_dict = actual_tree.to_dict()

        expected_dict = expected_tree.to_dict()

        self.assertDictEqual(actual_dict, expected_dict)

    def test_cherry_tree_to_dict_representation(self):

        expected_tree = self.build_cherry_tree(OldTree)

        actual_tree = self.build_cherry_tree(Tree)

        actual_dict = actual_tree.to_dict()

        expected_dict = expected_tree.to_dict()

        self.assertEqual(actual_tree, expected_tree)

        self.assertDictEqual(actual_dict, expected_dict)

    def test_linear_tree_to_dict_representation(self):

        expected_tree = self.build_linear_tree(OldTree)

        actual_tree = self.build_linear_tree(Tree)

        actual_dict = actual_tree.to_dict()

        expected_dict = expected_tree.to_dict()

        self.assertEqual(actual_tree, expected_tree)

        self.assertDictEqual(actual_dict, expected_dict)

    def test_single_node_tree_from_dict_representation(self):
        n = 100
        p = 1.0

        data = self._create_data_points(4, n, p)

        expected_tree = OldTree.get_single_node_tree(data)

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

    def test_cherry_tree_from_dict_representation(self):
        data = self.build_six_datapoints()

        expected_tree = self.build_cherry_tree(OldTree, data)

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

    def test_linear_tree_from_dict_representation(self):

        data = self.build_six_datapoints()

        expected_tree = self.build_linear_tree(OldTree, data)

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

    def test_getting_subtree_from_linear(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        expected_tree_roots = expected_tree.roots

        exp_n_1 = expected_tree.create_root_node(children=expected_tree_roots, data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=[exp_n_1], data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_1)

        exp_subtree = expected_tree.get_subtree(exp_n_1)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)

    def test_getting_subtree_from_cherry(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        exp_n_1 = expected_tree.create_root_node(children=[], data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=expected_tree_roots, data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_2)

        exp_subtree = expected_tree.get_subtree(exp_n_2)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)


    def test_removing_subtree_from_linear(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        expected_tree_roots = expected_tree.roots

        exp_n_1 = expected_tree.create_root_node(children=expected_tree_roots, data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=[exp_n_1], data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_1)

        exp_subtree = expected_tree.get_subtree(exp_n_1)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)

        actual_tree.remove_subtree(subtree)

        expected_tree.remove_subtree(exp_subtree)

        self.assertTrue(tree_eq(actual_tree, expected_tree))


    def test_removing_subtree_from_cherry(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        exp_n_1 = expected_tree.create_root_node(children=[], data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=expected_tree_roots, data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_2)

        exp_subtree = expected_tree.get_subtree(exp_n_2)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)

        actual_tree.remove_subtree(subtree)

        expected_tree.remove_subtree(exp_subtree)

        self.assertTrue(tree_eq(actual_tree, expected_tree))

    def test_adding_subtree_to_linear(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        expected_tree_roots = expected_tree.roots

        exp_n_1 = expected_tree.create_root_node(children=expected_tree_roots, data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=[exp_n_1], data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_1)

        exp_subtree = expected_tree.get_subtree(exp_n_1)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)

        actual_tree.remove_subtree(subtree)

        expected_tree.remove_subtree(exp_subtree)

        self.assertTrue(tree_eq(actual_tree, expected_tree))

        actual_tree.add_subtree(subtree, exp_n_2)

        expected_tree.add_subtree(exp_subtree, exp_n_2)

        self.assertTrue(tree_eq(actual_tree, expected_tree))

    def test_adding_subtree_to_cherry(self):
        data = self.build_six_datapoints()

        expected_tree = OldTree.get_single_node_tree(data[:2])

        exp_n_1 = expected_tree.create_root_node(children=[], data=data[2:4])

        expected_tree_roots = expected_tree.roots

        exp_n_2 = expected_tree.create_root_node(children=expected_tree_roots, data=data[4:])

        expected_dict = expected_tree.to_dict()

        actual_tree = Tree.from_dict(expected_dict)

        self.assertEqual(expected_tree, actual_tree)

        subtree = actual_tree.get_subtree(exp_n_2)

        exp_subtree = expected_tree.get_subtree(exp_n_2)

        actual_dict2 = subtree.to_dict()

        expected_dict2 = exp_subtree.to_dict()

        self.assertDictEqual(actual_dict2, expected_dict2)

        actual_tree.remove_subtree(subtree)

        expected_tree.remove_subtree(exp_subtree)

        self.assertTrue(tree_eq(actual_tree, expected_tree))

        actual_tree.add_subtree(subtree, None)

        expected_tree.add_subtree(exp_subtree, None)

        self.assertTrue(tree_eq(actual_tree, expected_tree))

    def _create_data_point(self, idx, n, p):
        return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

    def _create_data_points(self, size, n, p, start_idx=0):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i+start_idx, n, p))

        return result


def tree_eq(self_tree, other):
    self_key = (get_clades(self_tree), frozenset(self_tree.outliers))

    other_key = (get_clades(other), frozenset(other.outliers))

    return self_key == other_key
