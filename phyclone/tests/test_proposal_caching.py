import unittest
import numpy as np

from phyclone.smc.swarm import Particle
from phyclone.tree import TreeJointDistribution, FSCRPDistribution, Tree
from phyclone.smc.kernels.fully_adapted import _get_cached_full_proposal_dist
from phyclone.smc.kernels.semi_adapted import _get_cached_semi_proposal_dist
from phyclone.smc.kernels import FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.tests.simulate import simulate_binomial_data


class BaseTest(object):
    class BaseTest(unittest.TestCase):
        kernel = None
        cached_fxn = None
        outlier_prob = 0.0
        n = 100
        p = 1.0

        def __init__(self, method_name: str = ...):
            super().__init__(method_name)

            self._rng = np.random.default_rng(12345)

        def tearDown(self):
            self.cached_fxn.cache_clear()

        def _get_sampler(self, kernel_cls):

            self.tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

            kernel = kernel_cls(self.tree_dist, outlier_proposal_prob=self.outlier_prob, perm_dist=None,
                                rng=self._rng)

            return kernel

        def _create_data_point(self, idx, n, p):
            return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

        def _create_data_points(self, size, n, p, start_idx=0):
            result = []

            for i in range(size):
                result.append(self._create_data_point(i + start_idx, n, p))

            return result

        def test_no_parent_tree_or_particle(self):
            datapoint = self._create_data_point(0, self.n, self.p)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 0)

            prop_1 = self.kernel.get_proposal_distribution(datapoint, None, None)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 1)

            prop_2 = self.kernel.get_proposal_distribution(datapoint, None, None)

            self.assertEqual(prop_2, prop_1)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 1)
            self.assertEqual(cache_size, 1)

        def test_no_parent_tree_or_particle_different_alphas(self):
            datapoint = self._create_data_point(0, self.n, self.p)

            cache_size, num_hits = self.get_cache_info()

            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 0)

            prop_1 = self.kernel.get_proposal_distribution(datapoint, None, None)

            cache_size, num_hits = self.get_cache_info()

            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 1)

            self.kernel.tree_dist.prior.alpha = 2.0

            prop_2 = self.kernel.get_proposal_distribution(datapoint, None, None)

            self.assertNotEqual(prop_1, prop_2)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 2)

        def test_single_node_parent_same_alpha(self):
            data = self._create_data_points(4, self.n, self.p)
            tree_data = data[:-1]
            datapoint = data[-1]

            parent_tree = Tree.get_single_node_tree(tree_data)

            parent_particle = Particle(0, None, parent_tree, self.tree_dist, self.kernel.perm_dist)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 0)

            prop_1 = self.kernel.get_proposal_distribution(datapoint, parent_particle, parent_tree)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 1)

            prop_2 = self.kernel.get_proposal_distribution(datapoint, parent_particle, parent_tree)

            self.assertEqual(prop_2, prop_1)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 1)
            self.assertEqual(cache_size, 1)

        def test_single_node_parent_same_alpha_equivalent_parent_particles(self):
            data = self._create_data_points(4, self.n, self.p)
            tree_data = data[:-1]
            datapoint = data[-1]

            parent_tree = Tree.get_single_node_tree(tree_data)

            parent_particle = Particle(0, None, parent_tree, self.tree_dist, self.kernel.perm_dist)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 0)

            prop_1 = self.kernel.get_proposal_distribution(datapoint, parent_particle, parent_tree)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 1)

            parent_tree2 = Tree.get_single_node_tree(tree_data)

            parent_particle2 = Particle(0, None, parent_tree2, self.tree_dist, self.kernel.perm_dist)

            prop_2 = self.kernel.get_proposal_distribution(datapoint, parent_particle2, parent_tree)

            self.assertEqual(prop_2, prop_1)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 1)
            self.assertEqual(cache_size, 1)

        def test_single_node_parent_different_alpha(self):
            data = self._create_data_points(4, self.n, self.p)
            tree_data = data[:-1]
            datapoint = data[-1]

            parent_tree = Tree.get_single_node_tree(tree_data)

            parent_particle = Particle(0, None, parent_tree, self.tree_dist, self.kernel.perm_dist)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 0)

            prop_1 = self.kernel.get_proposal_distribution(datapoint, parent_particle, parent_tree)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 1)

            self.kernel.tree_dist.prior.alpha = 2.0

            prop_2 = self.kernel.get_proposal_distribution(datapoint, parent_particle, parent_tree)

            self.assertNotEqual(prop_1, prop_2)

            cache_size, num_hits = self.get_cache_info()
            self.assertEqual(num_hits, 0)
            self.assertEqual(cache_size, 2)

        def get_cache_info(self):
            num_hits = self.cached_fxn.cache_info().hits
            cache_size = self.cached_fxn.cache_info().currsize
            return cache_size, num_hits


class FullyAdaptedTest(BaseTest.BaseTest):

    def setUp(self):
        self.kernel = self._get_sampler(FullyAdaptedKernel)
        self.cached_fxn = _get_cached_full_proposal_dist


class SemiAdaptedTest(BaseTest.BaseTest):

    def setUp(self):
        self.kernel = self._get_sampler(SemiAdaptedKernel)
        self.cached_fxn = _get_cached_semi_proposal_dist


if __name__ == "__main__":
    unittest.main()
