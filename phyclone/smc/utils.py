'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict

import random

from phyclone.tree import Tree


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

    return Tree(list(nodes.values()))


def sample_sigma(tree, source=None):
    if source is None:
        sigma = []

        for node in tree.roots:
            sigma.append(sample_sigma(tree, source=node))

        return interleave_lists(sigma)

    child_sigma = []

    children = tree.get_children_nodes(source)

    random.shuffle(children)

    for child in children:
        child_sigma.append(sample_sigma(tree, source=child))

    sigma = interleave_lists(child_sigma)

    source_sigma = list([x.idx for x in tree.nodes[source.idx].data])

    random.shuffle(source_sigma)

    sigma.extend(source_sigma)

    return sigma


def interleave_lists(lists):
    result = []

    while len(lists) > 0:
        x = random.choice(lists)

        result.append(x.pop(0))

        if len(x) == 0:
            lists.remove(x)

    return result
