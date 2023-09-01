import unittest

import numpy as np

from phyclone.data.base import DataPoint
from phyclone.tree import FSCRPDistribution, Tree
from phyclone.math_utils import log_factorial

from math import inf
from phyclone.math_utils import simple_log_factorial
from numpy import full


class Test(unittest.TestCase):

    def setUp(self):
        grid_size = (1, 10)

        factorial_arr = full(8, -inf)
        simple_log_factorial(7, factorial_arr)

        memo_logs = {"log_p": {}, "log_r": {}, "log_s": {}}

        self.tree = Tree(grid_size, factorial_arr, memo_logs)
        
        self.tree_dist = FSCRPDistribution(1.0)

    def test_create_root_node(self):
        node = self.tree.create_root_node([])

        self.assertEqual(node, 0)

        self.assertListEqual(self.tree.nodes, [0])

        self.assertListEqual(self.tree.roots, [0])

    def test_simple_log_p(self):
        self.assertEqual(self.tree.log_p_one, 0)

    def test_one_data_point_sigma(self):
        data_point = self._create_data_point(0)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data_point, node)

        self.assertEqual(self.tree_dist.log_p(self.tree), 0)

    def test_two_data_point_one_cluster_sigma(self):
        """ CRP term is $\alpha * log(x - 1) = 1.0 * log(2)! and tree term is (n+1)^(n-1) = 2^0
        """
        data = self._create_data_points(3)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        self.tree.add_data_point_to_node(data[1], node)
        
        self.tree.add_data_point_to_node(data[2], node)

        self.assertEqual(self.tree_dist.log_p(self.tree), log_factorial(2))

    def test_two_data_point_chain_sigma(self):
        """ CRP is 0 and tree term is (n+1)^(n-1) = 3^1
        """
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([node])

        self.tree.add_data_point_to_node(data[1], node)

        self.assertEqual(self.tree_dist.log_p(self.tree), -np.log(3))

    def test_two_data_point_separate_sigma(self):
        """ CRP is 0 and tree term is (n+1)^(n-1) = 3^1
        """
        data = self._create_data_points(2)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[0], node)

        node = self.tree.create_root_node([])

        self.tree.add_data_point_to_node(data[1], node)

        self.assertEqual(self.tree_dist.log_p(self.tree), -np.log(3))

    def _create_data_point(self, idx):
        return DataPoint(idx, np.zeros(self.tree.grid_size))

    def _create_data_points(self, size):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i))

        return result
