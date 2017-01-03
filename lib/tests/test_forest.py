'''
Created on 2014-02-22

@author: andrew
'''
from __future__ import division

import unittest

from fscrp.forest import Forest, Node

class Test(unittest.TestCase):
    def test_node_counts(self):
        a = Node(sum, [], 0)
        
        b = Node(sum, [], 0)
        
        c = Node(sum, [a, b], 0)
        
        forest = Forest()
        
        forest.add_node(a)
        
        forest.add_node(b)
        
        forest.add_node(c)
        
        self.assertEqual(len(forest.nodes), 3)
        
        self.assertEqual(len(forest.root_nodes), 1)
        
    def test_node_agg_params(self):
        a = Node(sum, [], 1)
        
        b = Node(sum, [], 2)
        
        c = Node(sum, [a, b], 3)
        
        self.assertEqual(a.node_param, 1)
        
        self.assertEqual(a.agg_param, 1)
        
        self.assertEqual(c.node_param, 3)
        
        self.assertEqual(c.agg_param, 6)
        
    def test_node_agg_params_mean(self):
        a = Node(mean, [], 1)
        
        b = Node(mean, [], 2)
        
        c = Node(mean, [a, b], 3)
        
        self.assertEqual(a.node_param, 1)
        
        self.assertEqual(a.agg_param, 1)
        
        self.assertEqual(c.node_param, 3)
        
        self.assertEqual(c.agg_param, 2)           

def mean(x):
    return sum(x) / len(x)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()