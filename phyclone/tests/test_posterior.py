import unittest

from collections import defaultdict, Counter

import numpy as np

from phyclone.consensus import get_clades
from phyclone.data import DataPoint
from phyclone.mcmc import ParticleGibbsTreeSampler
from phyclone.tree import Tree
from phyclone.tests.exact_posterior import get_exact_posterior
from phyclone.tests.toy_data import load_test_data


class Test(unittest.TestCase):

    def test_particle_gibbs(self):
        #         data = self._get_no_data_data(2)

        data = load_test_data(2, depth=int(1e6), single_sample=True)

        sampler = ParticleGibbsTreeSampler('fully-adapted')

        pred_probs = self._run_no_data_test(data, sampler, burnin=100, num_iters=1000)

        true_probs = get_exact_posterior(data)

        self._test_posterior(pred_probs, true_probs)

    def _run_no_data_test(self, data, sampler, burnin=0, num_iters=int(1e3)):

        test_counts = Counter()

        tree = Tree.get_single_node_tree(data)

        for i in range(-burnin, num_iters):
            if i % 10 == 0:
                print(i)

            tree = sampler.sample_tree(data, tree)

            if i > 0:
                test_counts[get_clades(tree)] += 1

        norm_const = sum(test_counts.values())

        posterior_probs = defaultdict(float)

        for key in test_counts:
            posterior_probs[key] = test_counts[key] / norm_const

        return posterior_probs

    def _get_no_data_data(self, num_data_points, grid_size=(100, 3)):
        data = []

        for idx in range(num_data_points):
            data.append(DataPoint(idx, np.zeros(grid_size), 0.0))

        return data

    def _test_posterior(self, pred_probs, true_probs):
        print(list(pred_probs.items()))
        print()
        print(list(true_probs.items()))
        for key in true_probs:
            self.assertAlmostEqual(pred_probs[key], true_probs[key], delta=0.02)


if __name__ == "__main__":
    unittest.main()
