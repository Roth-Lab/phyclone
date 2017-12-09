'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict

import networkx as nx
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
    return last_particle.state.nodes


def get_tree(particle, sigma=None):
    nodes = get_nodes(particle)

    return Tree(nodes.values())


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


def get_constrained_path(data, kernel, sigma, tree):
    constrained_path = [None, ]

    data_to_node = tree.labels

    node_idx = 0

    old_to_new_node_idx = {}

    root_idxs = set()

    for data_idx in sigma:
        old_node_idx = data_to_node[data_idx]

        if old_node_idx not in old_to_new_node_idx:
            for child in tree.nodes[old_node_idx].children:
                root_idxs.remove(old_to_new_node_idx[child.idx])

            old_to_new_node_idx[old_node_idx] = node_idx

            root_idxs.add(node_idx)

            node_idx += 1

        proposal_dist = kernel.get_proposal_distribution(data[data_idx], constrained_path[-1])

        state = kernel.create_state(data[data_idx], constrained_path[-1], old_to_new_node_idx[old_node_idx], root_idxs)

        log_q = proposal_dist.get_log_q(state)

        particle = kernel.create_particle(data[data_idx], log_q, constrained_path[-1], state)

        constrained_path.append(particle)

    assert nx.is_isomorphic(tree.graph, get_tree(constrained_path[-1], sigma).graph)

    return constrained_path
