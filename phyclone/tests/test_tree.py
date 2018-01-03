import unittest

import numpy as np

from phyclone.data import DataPoint
from phyclone.tree import Tree
from phyclone.math_utils import log_factorial


class Test(unittest.TestCase):

    def test_one_data_point_sigma(self):
        grid_size = (1, 10)

        data_point = DataPoint(0, np.zeros(grid_size))

        tree = Tree(1.0, grid_size)

        node = tree.create_root_node([])

        tree.add_data_point(data_point, node)

        self.assertEqual(tree.log_p_sigma, 0)

    def test_two_data_point_one_cluster_sigma(self):
        grid_size = (1, 10)

        data = [DataPoint(0, np.zeros(grid_size)), DataPoint(1, np.zeros(grid_size))]

        tree = Tree(1.0, grid_size)

        node = tree.create_root_node([])

        tree.add_data_point(data[0], node)

        tree.add_data_point(data[1], node)

        self.assertEqual(tree.log_p_sigma, -log_factorial(2))

    def test_two_data_point_chain_sigma(self):
        grid_size = (1, 10)

        data = [DataPoint(0, np.zeros(grid_size)), DataPoint(1, np.zeros(grid_size))]

        tree = Tree(1.0, grid_size)

        node = tree.create_root_node([])

        tree.add_data_point(data[0], node)

        node = tree.create_root_node([node, ])

        tree.add_data_point(data[1], node)

        self.assertEqual(tree.log_p_sigma, 0)

    def test_two_data_point_separate_sigma(self):
        grid_size = (1, 10)

        data = [DataPoint(0, np.zeros(grid_size)), DataPoint(1, np.zeros(grid_size))]

        tree = Tree(1.0, grid_size)

        node = tree.create_root_node([])

        tree.add_data_point(data[0], node)

        node = tree.create_root_node([])

        tree.add_data_point(data[1], node)

        self.assertEqual(tree.log_p_sigma, -log_factorial(2))
