'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict
from scipy.misc import logsumexp as log_sum_exp

import numpy as np
import random

from phyclone.tree import Tree


class Configuration(object):
    def __init__(self, outliers, tree):
        self.outliers = outliers

        self.tree = tree

    @property
    def labels(self):
        labels = self.tree.labels

        for data_point in self.outliers:
            labels[data_point.idx] = -1

        return labels

    @property
    def log_p(self):
        log_p = self.tree.log_p

        for data_point in self.outliers:
            log_norm = np.log(data_point.value.shape[1])

            log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return log_p

    @property
    def log_p_one(self):
        log_p = self.tree.log_p_one

        for data_point in self.outliers:
            log_norm = np.log(data_point.value.shape[1])

            log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return log_p


def iter_particles(particle):
    while particle is not None:
        yield particle

        particle = particle.parent_particle


def get_num_data_points_per_node(last_particle):
    counts = defaultdict(int)

    for particle in iter_particles(last_particle):
        counts[particle.state.node_idx] += 1

    return dict(counts)


def get_nodes(last_particle):
    nodes = last_particle.state.roots

    for particle in iter_particles(last_particle):
        for node_idx in particle.state.roots:
            if node_idx not in nodes:
                nodes[node_idx] = particle.state.roots[node_idx]

    return nodes


def get_tree(particle):
    nodes = get_nodes(particle)

    tree = Tree(list(nodes.values()))

    return Configuration(particle.state.outliers, tree)


def sample_sigma(config, source=None):
    if source is None:
        sigma = []

        for node in config.tree.roots:
            sigma.append(sample_sigma(config, source=node))

        sigma.append([x.idx for x in config.outliers])

        return interleave_lists(sigma)

    child_sigma = []

    children = config.tree.get_children_nodes(source)

    random.shuffle(children)

    for child in children:
        child_sigma.append(sample_sigma(config, source=child))

    sigma = interleave_lists(child_sigma)

    source_sigma = list([x.idx for x in config.tree.nodes[source.idx].data])

    random.shuffle(source_sigma)

    sigma.extend(source_sigma)

    return sigma


def interleave_lists(lists):
    result = []

    while len(lists) > 0:
        x = random.choice(lists)

        if len(x) == 0:
            lists.remove(x)

        else:
            result.append(x.pop(0))

    return result
