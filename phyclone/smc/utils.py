'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import random

from phyclone.math_utils import log_factorial, log_binomial_coefficient, log_multinomial_coefficient


class NodePermutationDistribution(object):
    @staticmethod
    def log_count(tree, source=None):
        if source is None:
            num_data_points = len(tree.data)

            num_nodes = len(tree.nodes)

            num_outliers = len(tree.outliers)

            num_others = num_data_points - num_nodes - num_outliers

            # Count assignment of single data point for each node
            count = 0

            root_counts = []

            for node in tree.roots:
                count += NodePermutationDistribution.log_count(tree, source=node)

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
            count += NodePermutationDistribution.log_count(tree, source=child)

            child_counts.append(len(tree.get_descendants(child)) + 1)

        # Bridge shuffle of children
        count += log_factorial(sum(child_counts))

        for c in child_counts:
            count -= log_factorial(c)

        # Select point from source
        count += np.log(len(tree.get_data(source)))

        return count

    @staticmethod
    def log_pdf(tree):
        return -NodePermutationDistribution.log_count(tree)

    @staticmethod
    def sample(tree, source=None):
        if source is None:
            sigma = []

            for node in tree.roots:
                sigma.append(NodePermutationDistribution.sample(tree, source=node))

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
            sigma.append(NodePermutationDistribution.sample(tree, source=child))

        sigma = interleave_lists(sigma)

        source_data = tree.get_data(source)

        sigma.append(random.choice(source_data))

        return sigma


class RootPermutationDistribution(object):
    @staticmethod
    def log_count(tree, source=None):
        if source is None:
            count = 0

            subtree_sizes = []

            roots = tree.roots

            for node in roots:
                count += RootPermutationDistribution.log_count(tree, source=node)

                subtree_sizes.append(len(tree.get_subtree_data(node)))

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

                subtree_sizes.append(len(tree.get_subtree_data(child)))

            # Bridge shuffle
            count += log_multinomial_coefficient(subtree_sizes)

            # Permute the source data
            count += log_factorial(len(tree.get_data(source)))

        return count

    @staticmethod
    def log_pdf(tree):
        return -RootPermutationDistribution.log_count(tree)

    @staticmethod
    def sample(tree, source=None):
        if source is None:
            sigma = []

            roots = tree.roots

            for node in roots:
                sigma.append(RootPermutationDistribution.sample(tree, source=node))

            # Bridge shuffle root lists
            sigma = interleave_lists(sigma)

            # Bridge shuffle outliers and tree data
            outliers = list(tree.outliers)

            random.shuffle(outliers)

            sigma = interleave_lists([sigma, outliers])

        else:
            child_sigma = []

            children = tree.get_children(source)

            for child in children:
                child_sigma.append(RootPermutationDistribution.sample(tree, source=child))

            # Bridge shuffle
            sigma = interleave_lists(child_sigma)

            # Permute source data
            source_sigma = tree.get_data(source)

            random.shuffle(source_sigma)

            sigma.extend(source_sigma)

        return sigma


def interleave_lists(lists):
    result = []

    sentinels = []

    for i, l in enumerate(lists):
        sentinels.extend(np.ones(len(l), dtype=int) * i)

    random.shuffle(sentinels)

    for idx in sentinels:
        result.append(lists[idx].pop(0))

    return result
