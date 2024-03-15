'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
import numpy as np
from phyclone.utils.math import log_factorial, log_binomial_coefficient, log_multinomial_coefficient


class RootPermutationDistribution(object):

    @staticmethod
    def log_count(tree, source=None):
        if source is None:
            count = 0

            subtree_sizes = []

            roots = tree.roots

            for node in roots:
                count += RootPermutationDistribution.log_count(tree, source=node)

                # subtree_sizes.append(len(tree.get_subtree_data(node)))
                subtree_sizes.append(tree.get_subtree_data_len(node))

            # Bridge shuffle root nodes
            count += log_multinomial_coefficient(subtree_sizes)

            num_data_points = len(tree.data)

            num_outlier_data_points = len(tree.outliers)

            # Bridge shuffle outliers
            count += log_binomial_coefficient(num_data_points, num_outlier_data_points)

        else:
            count = 0

            subtree_sizes = []

            children = tree.get_children(source)

            for child in children:
                count += RootPermutationDistribution.log_count(tree, source=child)

                # subtree_sizes.append(len(tree.get_subtree_data(child)))
                subtree_sizes.append(tree.get_subtree_data_len(child))

            # Bridge shuffle
            count += log_multinomial_coefficient(subtree_sizes)

            # Permute the source data
            # count += log_factorial(len(tree.get_data(source)))
            count += log_factorial(tree.get_data_len(source))
            # count += cached_log_factorial(tree.get_data_len(source))

        return count

    @staticmethod
    def log_pdf(tree):
        return -RootPermutationDistribution.log_count(tree)

    @staticmethod
    def sample(tree, rng, source=None):
        if source is None:
            sigma = []

            roots = tree.roots

            for node in roots:
                sigma.append(RootPermutationDistribution.sample(tree, rng, source=node))

            # Bridge shuffle root lists
            sigma = interleave_lists(sigma, rng)

            # Bridge shuffle outliers and tree data
            outliers = list(tree.outliers)

            # random.shuffle(outliers)
            rng.shuffle(outliers)

            sigma = interleave_lists([sigma, outliers], rng)

        else:
            child_sigma = []

            children = tree.get_children(source)

            for child in children:
                child_sigma.append(RootPermutationDistribution.sample(tree, rng, source=child))

            # Bridge shuffle
            sigma = interleave_lists(child_sigma, rng)

            # Permute source data
            source_sigma = tree.get_data(source)

            # random.shuffle(source_sigma)
            rng.shuffle(source_sigma)

            sigma.extend(source_sigma)

        return sigma


def interleave_lists(lists, rng):
    # result = []

    sentinels = []

    for i, l in enumerate(lists):
        sentinels.extend(np.ones(len(l), dtype=int) * i)

    # random.shuffle(sentinels)
    rng.shuffle(sentinels)

    # for idx in sentinels:
    #     result.append(lists[idx].pop(0))

    result = [lists[idx].pop(0) for idx in sentinels]

    return result
