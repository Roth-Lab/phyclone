import unittest

from collections import defaultdict, Counter

from phyclone.consensus import get_clades
from phyclone.mcmc import ParticleGibbsTreeSampler
from phyclone.smc.kernels import BootstrapKernel, FullyAdaptedKernel, SemiAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.tests.exact_posterior import get_exact_posterior
from phyclone.smc.utils import RootPermutationDistribution

import phyclone.tests.simulate as simulate

from math import inf
from phyclone.math_utils import simple_log_factorial
from numpy import full


class BaseTest(object):

    class BaseTest(unittest.TestCase): 

        def __init__(self, methodName: str = ...):
            super().__init__(methodName)
            self.factorial_arr = None

            self.memo_logs = None

        def test_single_data_point_1d(self):
            data = [simulate.simulate_binomial_data(0, 100, 1.0), ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=100)
    
        def test_single_data_point_2d(self):
            data = [simulate.simulate_binomial_data(0, 100, [1.0, 1.0]), ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=100)
    
        def test_two_data_point_1d_non_informative(self):
            data = [
                simulate.simulate_binomial_data(0, 0, 1.0),
                simulate.simulate_binomial_data(1, 0, 1.0)
            ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=2000)
    
        def test_three_data_point_1d_non_informative(self):
            data = [
                simulate.simulate_binomial_data(0, 0, 1.0),
                simulate.simulate_binomial_data(1, 0, 1.0),
                simulate.simulate_binomial_data(2, 0, 1.0),
            ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=2000)
    
        def test_two_data_point_1d_single_cluster(self):
            data = [
                simulate.simulate_binomial_data(0, 10, 1.0),
                simulate.simulate_binomial_data(1, 10, 1.0)
            ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=2000)
    
        def test_two_data_point_1d_two_cluster(self):
            data = [
                simulate.simulate_binomial_data(0, 10, 1.0),
                simulate.simulate_binomial_data(1, 10, 0.5)
            ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=2000)
    
        def test_two_data_point_2d_two_cluster(self):
            data = [
                simulate.simulate_binomial_data(0, 100, [1.0, 1.0]),
                simulate.simulate_binomial_data(1, 100, [0.5, 0.7])
            ]
    
            self._run_exact_posterior_test(data, burnin=100, num_iters=1000)
            
        def _get_sampler(self, kernel_cls):
            perm_dist = RootPermutationDistribution()
            
            self.tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

            factorial_arr = full(6, -inf)
            simple_log_factorial(5, factorial_arr)

            memo_logs = {"log_p": {}, "log_r": {}, "log_s": {}}
            
            kernel = kernel_cls(self.tree_dist, outlier_proposal_prob=0, perm_dist=perm_dist,
                                factorial_arr=factorial_arr, memo_logs=memo_logs)

            self.factorial_arr = factorial_arr

            self.memo_logs = memo_logs
            
            return ParticleGibbsTreeSampler(kernel)            
    
        def _run_exact_posterior_test(self, data, burnin=100, num_iters=1000):
            pred_probs = self._run_sampler(data, burnin=burnin * self.run_scale, num_iters=num_iters * self.run_scale)
    
            true_probs = get_exact_posterior(data, self.tree_dist, self.factorial_arr, self.memo_logs)
    
            self._test_posterior(pred_probs, true_probs)
    
        def _run_sampler(self, data, burnin=0, num_iters=int(1e3)):
    
            test_counts = Counter()
    
            tree = Tree.get_single_node_tree(data, self.factorial_arr, self.memo_logs)
    
            for i in range(-burnin, num_iters):
                if i % 10 == 0:
                    print(i)
    
                tree = self.sampler.sample_tree(tree)
    
                if i > 0:
                    test_counts[get_clades(tree)] += 1
    
            norm_const = sum(test_counts.values())
    
            posterior_probs = defaultdict(float)
    
            for key in test_counts:
                posterior_probs[key] = test_counts[key] / norm_const
    
            return posterior_probs
    
        def _test_posterior(self, pred_probs, true_probs):
            print(list(pred_probs.items()))
            print()
            print(list(true_probs.items()))
            for key in true_probs:
                self.assertAlmostEqual(pred_probs[key], true_probs[key], delta=0.02)


class BootstrapAdaptedTest(BaseTest.BaseTest):

    def setUp(self):
        self.sampler = self._get_sampler(BootstrapKernel)
        
        self.run_scale = 1


class FullyAdaptedTest(BaseTest.BaseTest):
 
    def setUp(self):
        self.sampler = self._get_sampler(FullyAdaptedKernel)
         
        self.run_scale = 1


class SemiAdaptedTest(BaseTest.BaseTest):
 
    def setUp(self):
        self.sampler = self._get_sampler(SemiAdaptedKernel)
         
        self.run_scale = 1


if __name__ == "__main__":
    unittest.main()
