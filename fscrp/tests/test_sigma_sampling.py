'''
Created on 17 Mar 2017

@author: andrew
'''
import unittest

import networkx as nx

from fscrp.graph_utils import interleave_lists, sample_sigma


class Test(unittest.TestCase):

    def test_interleave(self):
        x = range(10)
        y = range(10, 20)

        z = interleave_lists([list(x), list(y)])

        prev_idx = -1

        for x_i in x:
            self.assertGreater(z.index(x_i), prev_idx)

            prev_idx = z.index(x_i)

        prev_idx = -1

        for y_i in y:
            self.assertGreater(z.index(y_i), prev_idx)

            prev_idx = z.index(y_i)

    def test_sample_sigma_tree(self):
        graph = nx.DiGraph()

        graph.add_node(0, data_points=range(10))

        graph.add_node(1, data_points=range(10, 20))

        graph.add_node(2, data_points=range(20, 30))

        graph.add_edge(0, 1)

        graph.add_edge(0, 2)

        sigma = sample_sigma(graph)

        self.assertGreater(min(sigma[:20]), 9)

    def test_sample_sigma_chain(self):
        graph = nx.DiGraph()

        graph.add_node(0, data_points=range(10))

        graph.add_node(1, data_points=range(10, 20))

        graph.add_node(2, data_points=range(20, 30))

        graph.add_edge(0, 1)

        graph.add_edge(1, 2)

        sigma = sample_sigma(graph)

        self.assertGreater(min(sigma[:10]), 19)

        self.assertGreater(min(sigma[:20]), 9)

    def test_sample_two_roots(self):
        graph = nx.DiGraph()

        graph.add_node(0, data_points=range(10))

        graph.add_node(1, data_points=range(10, 20))

        graph.add_node(2, data_points=range(20, 30))

        graph.add_node(3, data_points=range(30, 40))

        graph.add_edge(0, 2)

        graph.add_edge(1, 3)

        sigma = sample_sigma(graph)

        print sigma

        self.assertGreater(min(sigma[:20]), 19)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_interleave']
    unittest.main()
