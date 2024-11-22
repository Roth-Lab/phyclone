"""
Created on 9 Aug 2017

@author: Andrew Roth
"""

from itertools import repeat

from phyclone.utils.math import (
    log_factorial,
    log_binomial_coefficient,
    log_multinomial_coefficient,
)


class RootPermutationDistribution(object):
    __slots__ = ()

    @staticmethod
    def log_count(tree, source=None):
        if source is None:
            count = 0

            subtree_sizes = []

            roots = tree.roots

            for node in roots:
                count += RootPermutationDistribution.log_count(tree, source=node)

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

                subtree_sizes.append(tree.get_subtree_data_len(child))

            # Bridge shuffle
            count += log_multinomial_coefficient(subtree_sizes)

            # Permute the source data
            count += log_factorial(tree.get_data_len(source))

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

            rng.shuffle(source_sigma)

            sigma.extend(source_sigma)

        return sigma


def interleave_lists(lists, rng):

    sentinels = []

    for i, l in enumerate(lists):
        sentinels.extend(repeat(i, len(l)))

    rng.shuffle(sentinels)

    result = [lists[idx].pop(0) for idx in sentinels]

    return result
