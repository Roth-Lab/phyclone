import unittest

import numpy as np

from phyclone.data import DataPoint
from phyclone.tree import Tree
from phyclone.math_utils import log_factorial


class Test(unittest.TestCase):
    def setUp(self):
        grid_size = (1, 10)

        self.tree = Tree(1.0, grid_size)

    def test_create_root_node(self):
        node = self.tree.create_root_node([])

        self.assertEqual(node, 0)

        self.assertListEqual(self.tree.nodes, [0])

        self.assertListEqual(self.tree.roots, [0])

    def test_simple_log_p(self):
        self.assertEqual(self.tree.log_p, 0)

    def test_one_data_point_log_p(self):
        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(self._create_data_point(0), node)

        np.testing.assert_equal(
            self.tree.data_log_likelihood, self.tree.data_log_likelihood.max(axis=1)[:, np.newaxis]
        )

    def test_one_data_point_sigma(self):
        data_point = self._create_data_point(0)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data_point, node)

        self.assertEqual(self.tree.log_p_sigma, 0)

    def test_two_data_point_one_cluster_sigma(self):
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        self.tree.add_data_point_to_node(data[1], node)

        self.assertEqual(self.tree.log_p_sigma, -log_factorial(2))

    def test_two_data_point_chain_sigma(self):
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([node, ])

        self.tree.add_data_point_to_node(data[1], node)

        self.assertEqual(self.tree.log_p_sigma, 0)

    def test_two_data_point_separate_sigma(self):
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[1], node)

        self.assertEqual(self.tree.log_p_sigma, -log_factorial(2))

    def _create_data_point(self, idx):
        return DataPoint(idx, np.zeros(self.tree.grid_size))

    def _create_data_points(self, size):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i))

        return result
