'''
Created on 3 Jan 2018

@author: andrew
'''
import unittest

import numpy as np

from phyclone.data import DataPoint
from phyclone.tree import Tree
from phyclone.consensus import get_clades


class Test(unittest.TestCase):

    def test_chain_error(self):
        grid_shape = (1, 10)

        tree = Tree(1.0, grid_shape)

        data = [
            DataPoint(0, np.zeros(grid_shape)),
            DataPoint(1, np.zeros(grid_shape)),
            DataPoint(2, np.zeros(grid_shape)),
        ]

        node = tree.create_root_node([])

        tree.add_data_point(data[0], node)

        node = tree.create_root_node([node])

        tree.add_data_point(data[1], node)

        node = tree.create_root_node([node])

        tree.add_data_point(data[2], node)

        subtree = tree.get_subtree(tree.nodes[0])

        print('x', [x.idx for x in tree.nodes[0].data])

        print('a', get_clades(tree))

        print('b', get_clades(subtree))

        print('y', [x.idx for x in tree.roots])

        tree.remove_subtree(subtree)

        print('y', [x.idx for x in tree.roots])

        print('c', get_clades(tree))

        get_clades(tree)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
