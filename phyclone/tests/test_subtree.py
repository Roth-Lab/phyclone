'''
Created on 3 Jan 2018

@author: andrew
'''
import unittest

import numpy as np

from phyclone.data.base import DataPoint
from phyclone.tree import Tree
from phyclone.consensus import get_clades
from math import inf
from phyclone.math_utils import simple_log_factorial
from numpy import full


class Test(unittest.TestCase):

    def test_chain_error(self):
        grid_shape = (1, 10)

        factorial_arr = full(6, -inf)
        simple_log_factorial(5, factorial_arr)
        # self.factorial_arr = factorial_arr

        memo_logs = {"log_p": {}, "log_r": {}, "log_s": {}}

        tree = Tree(grid_shape, factorial_arr, memo_logs)

        data = [
            DataPoint(0, np.zeros(grid_shape)),
            DataPoint(1, np.zeros(grid_shape)),
            DataPoint(2, np.zeros(grid_shape)),
        ]

        node = tree.create_root_node([])

        tree.add_data_point_to_node(data[0], node)

        node = tree.create_root_node([node])

        tree.add_data_point_to_node(data[1], node)

        node = tree.create_root_node([node])

        tree.add_data_point_to_node(data[2], node)

        subtree = tree.get_subtree(tree.nodes[0])

        print('x', [x.idx for x in tree.node_data[node]])

        print('a', get_clades(tree))

        print('b', get_clades(subtree))

        print('y', tree.roots)

        tree.remove_subtree(subtree)

        print('y', tree.roots)

        print('c', get_clades(tree))

        get_clades(tree)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
