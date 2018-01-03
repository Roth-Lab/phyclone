import unittest

from collections import defaultdict, Counter

from phyclone.consensus import get_clades
from phyclone.mcmc import ParticleGibbsTreeSampler
from phyclone.tree import Tree
from phyclone.tests.exact_posterior import get_exact_posterior

import phyclone.tests.simulate as simulate


class Test(unittest.TestCase):

    def setUp(self):
        self.sampler = ParticleGibbsTreeSampler('semi-adapted')

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

    def _run_exact_posterior_test(self, data, burnin=100, num_iters=1000):
        pred_probs = self._run_sampler(data, burnin=burnin, num_iters=num_iters)

        true_probs = get_exact_posterior(data)

        self._test_posterior(pred_probs, true_probs)

    def _run_sampler(self, data, burnin=0, num_iters=int(1e3)):

        test_counts = Counter()

        tree = Tree.get_single_node_tree(data)

        for i in range(-burnin, num_iters):
            if i % 10 == 0:
                print(i)

            tree = self.sampler.sample_tree(data, tree)

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


if __name__ == "__main__":
    unittest.main()
