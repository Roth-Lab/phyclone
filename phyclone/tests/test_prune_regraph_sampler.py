import unittest
from unittest import mock

import numpy as np
from unittest.mock import patch, Mock

import phyclone.mcmc
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.tests.simulate import simulate_binomial_data
from phyclone.mcmc import PruneRegraphSampler


class TestPruneRegraph(unittest.TestCase):

    def setUp(self):

        self.outlier_prob = 0.0

        self._rng = np.random.default_rng(12345)

        tree_sub_dist = FSCRPDistribution(1.0)

        self.tree_dist = TreeJointDistribution(tree_sub_dist)

        self.prg_sampler = PruneRegraphSampler(self.tree_dist, self._rng)

    def test_single_node_tree(self):
        n = 100
        p = 1.0

        data = self._create_data_points(4, n, p)

        tree = Tree.get_single_node_tree(data)

        actual = self.prg_sampler.sample_tree(tree)

        self.assertEqual(tree, actual)

    # def test_two_node_linear_tree(self):
    #     n = 100
    #     p = 1.0
    #
    #     data = self._create_data_points(4, n, p)
    #
    #     tree = Tree.get_single_node_tree(data[:2])
    #
    #     tree_roots = tree.roots
    #
    #     n_1 = tree.create_root_node(children=tree_roots, data=data[2:])
    #
    #     rng_mock = mock.Mock(spec=np.random.Generator)
    #     rng_mock.choice.return_value = 0
    #     prg_sampler = PruneRegraphSampler(self.tree_dist, rng_mock)
    #
    #     remaining_nodes, subtree_root = prg_sampler._get_subtree_and_pruned_tree(tree)
    #
    #     with patch.object(self.prg_sampler, '_get_subtree_and_pruned_tree') as mock_sampler:
    #         mock_sampler.return_value = remaining_nodes, subtree_root
    #
    #         trees = prg_sampler._create_sampled_trees_array(remaining_nodes, subtree_root, tree)
    #
    #         sampled = self.prg_sampler.sample_tree(tree)
    #
    #     print('')

    def _create_data_point(self, idx, n, p):
        return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

    def _create_data_points(self, size, n, p, start_idx=0):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i+start_idx, n, p))

        return result
