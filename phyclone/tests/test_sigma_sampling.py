"""
Created on 17 Mar 2017

@author: andrew
"""

import unittest

import numpy as np

from phyclone.data.base import DataPoint
from phyclone.smc.utils import interleave_lists, RootPermutationDistribution
from phyclone.tree import Tree


class Test(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self._rng = np.random.default_rng(12345)

    def test_interleave(self):
        x = range(10)
        y = range(10, 20)

        z = interleave_lists([list(x), list(y)], self._rng)

        prev_idx = -1

        for x_i in x:
            self.assertGreater(z.index(x_i), prev_idx)

            prev_idx = z.index(x_i)

        prev_idx = -1

        for y_i in y:
            self.assertGreater(z.index(y_i), prev_idx)

            prev_idx = z.index(y_i)

    def test_sample_sigma_tree(self):
        grid_size = (1, 10)

        tree = Tree(grid_size)

        node_1 = tree.create_root_node(children=[], data=[])

        for i in range(10, 20):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_1)

        node_2 = tree.create_root_node(children=[], data=[])

        for i in range(20, 30):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_2)

        root = tree.create_root_node(children=[node_1, node_2], data=[])

        for i in range(10):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), root)

        sigma = RootPermutationDistribution.sample(tree, self._rng)

        self.assertGreater(min([x.idx for x in sigma[:20]]), 9)

    def test_sample_sigma_chain(self):
        grid_size = (1, 10)

        tree = Tree(grid_size)

        node_2 = tree.create_root_node(children=[], data=[])

        for i in range(20, 30):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_2)

        node_1 = tree.create_root_node(children=[node_2], data=[])

        for i in range(10, 20):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_1)

        root = tree.create_root_node(children=[node_1], data=[])

        for i in range(10):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), root)

        sigma = RootPermutationDistribution.sample(tree, self._rng)

        self.assertGreater(min([x.idx for x in sigma[:10]]), 19)

        self.assertGreater(min([x.idx for x in sigma[:20]]), 9)

    def test_sample_two_roots(self):
        grid_size = (1, 10)

        tree = Tree(grid_size)

        node_2 = tree.create_root_node(children=[], data=[])

        for i in range(20, 30):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_2)

        node_0 = tree.create_root_node(children=[node_2], data=[])

        for i in range(10):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_0)

        node_3 = tree.create_root_node(children=[], data=[])

        for i in range(30, 40):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_3)

        node_1 = tree.create_root_node(children=[node_3], data=[])

        for i in range(10, 20):
            tree.add_data_point_to_node(DataPoint(i, np.zeros(grid_size)), node_1)

        sigma = RootPermutationDistribution.sample(tree, self._rng)

        # self.assertGreater(min([x.idx for x in sigma[:20]]), 19)
        self.assertGreater(min([x.idx for x in sigma[:10]]), 19)
        self.assertLess(max([x.idx for x in sigma[30:]]), 20)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_interleave']
    unittest.main()
