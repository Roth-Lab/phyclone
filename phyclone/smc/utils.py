'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import random

from phyclone.math_utils import log_factorial


class PermutationDistribution(object):
    @staticmethod
    def count(tree, source=None):
        if source is None:
            num_data_points = len(tree.data)

            num_nodes = len(tree.nodes)

            num_outliers = len(tree.outliers)

            num_others = num_data_points - num_nodes - num_outliers

            # Count assignment of single data point for each node
            count = 0

            root_counts = []

            for node in tree.roots:
                count += PermutationDistribution.count(tree, source=node)

                root_counts.append(len(tree.get_descendants(node)) + 1)

            # Bridge shuffle of roots
            count += log_factorial(sum(root_counts))

            for c in root_counts:
                count -= log_factorial(c)

            # Permute outliers
            count += log_factorial(num_outliers)

            # Permute remaining nodes
            count += log_factorial(num_others)

            # Bridge shuffle of outliers
            count += log_factorial(num_data_points)
            count -= log_factorial(num_data_points - num_outliers)
            count -= log_factorial(num_outliers)

            return count

        count = 0

        children = tree.get_children(source)

        child_counts = []

        for child in children:
            count += PermutationDistribution.count(tree, source=child)

            child_counts.append(len(tree.get_descendants(child)) + 1)

        # Bridge shuffle of children
        count += log_factorial(sum(child_counts))

        for c in child_counts:
            count -= log_factorial(c)

        # Select point from source
        count += len(tree.get_data(source))

        return count

    @staticmethod
    def log_pdf(tree):
        return -PermutationDistribution.count(tree)

    @staticmethod
    def sample(tree, source=None):
        if source is None:
            sigma = []

            for node in tree.roots:
                sigma.append(PermutationDistribution.sample(tree, source=node))

            sigma = interleave_lists(sigma)

            outliers = list(tree.outliers)

            random.shuffle(outliers)

            others = [x for x in tree.data if (x not in sigma) and (x not in outliers)]

            random.shuffle(others)

            sigma.extend(others)

            sigma = interleave_lists([sigma, outliers])

            return sigma

        sigma = []

        children = tree.get_children(source)

        random.shuffle(children)

        for child in children:
            sigma.append(PermutationDistribution.sample(tree, source=child))

        sigma = interleave_lists(sigma)

        source_data = tree.get_data(source)

        sigma.append(random.choice(source_data))

        return sigma


# def sample_sigma(tree, source=None):
#     if source is None:
#         sigma = []
#
#         for node in tree.roots:
#             sigma.append(sample_sigma(tree, source=node))
#
#         outliers = list(tree.outliers)
#
#         random.shuffle(outliers)
#
#         sigma.append(outliers)
#
#         return interleave_lists(sigma)
#
#     child_sigma = []
#
#     children = tree.get_children(source)
#
#     random.shuffle(children)
#
#     for child in children:
#         child_sigma.append(sample_sigma(tree, source=child))
#
#     sigma = interleave_lists(child_sigma)
#
#     source_sigma = tree.get_data(source)
#
#     random.shuffle(source_sigma)
#
#     sigma.extend(source_sigma)
#
#     return sigma


def interleave_lists(lists):
    result = []

    while len(lists) > 0:
        x = random.choice(lists)

        if len(x) == 0:
            lists.remove(x)

        else:
            result.append(x.pop(0))

    return result
