'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict

import networkx as nx
import random

from fscrp.particle_utils import iter_particles
from fscrp.tree import Tree


def get_num_data_points_per_node(last_particle):
    counts = defaultdict(int)

    for particle in iter_particles(last_particle):
        counts[particle.state.node_idx] += 1

    return dict(counts)


def get_nodes(last_particle):
    return last_particle.state.nodes


def get_graph(particle, sigma=None):
    graph = nx.DiGraph()

    nodes = get_nodes(particle)

    node_data_points = get_node_data_points(particle, sigma=sigma)

    graph.add_node(
        -1,
        data_points=[],
        node=particle.state.dummy_root,
    )

    for idx in particle.state.root_idxs:
        graph.add_edge(-1, idx)

    for node_idx in node_data_points:
        graph.add_node(
            node_idx,
            data_points=node_data_points[node_idx],
            node=nodes[node_idx]
        )

    for node in nodes.values():
        for child in node.children:
            graph.add_edge(node.idx, child.idx)

    return graph


def get_tree(particle, sigma=None):
    data_points = get_node_data_points(particle, sigma=sigma)

    nodes = get_nodes(particle)

    return Tree(data_points, nodes.values())


def get_node_data_points(last_particle, sigma=None):
    node_data_points = defaultdict(list)

    for i, particle in enumerate(reversed(list(iter_particles(last_particle)))):
        if sigma is None:
            node_data_points[particle.state.node_idx].append(i)

        else:
            node_data_points[particle.state.node_idx].append(sigma[i])

    return node_data_points


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

    source_sigma = list(tree.data_points[source.idx])

    random.shuffle(source_sigma)

    sigma.extend(source_sigma)

    return sigma


# def sample_sigma(graph, source=None):
#     if source is None:
#         sigma = []
#
#         for node in graph.successors(-1):
#             sigma.append(sample_sigma(graph, source=node))
#
#         return interleave_lists(sigma)
#
#     child_sigma = []
#
#     children = list(graph.successors(source))
#
#     random.shuffle(children)
#
#     for child in children:
#         child_sigma.append(sample_sigma(graph, source=child))
#
#     sigma = interleave_lists(child_sigma)
#
#     source_sigma = list(graph.node[source]['data_points'])
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

# def get_constrained_path(data, graph, kernel, sigma):
#     constrained_path = [None, ]
#
#     data_to_node = get_data_to_node_map(graph)
#
#     node_idx = 0
#
#     old_to_new_node_idx = {}
#
#     root_idxs = set()
#
#     for data_idx in sigma:
#         old_node_idx = data_to_node[data_idx]
#
#         if old_node_idx not in old_to_new_node_idx:
#             for child_idx in graph.successors(old_node_idx):
#                 root_idxs.remove(old_to_new_node_idx[child_idx])
#
#             old_to_new_node_idx[old_node_idx] = node_idx
#
#             root_idxs.add(node_idx)
#
#             node_idx += 1
#
#         proposal_dist = kernel.get_proposal_distribution(data[data_idx], constrained_path[-1])
#
#         state = kernel.create_state(data[data_idx], constrained_path[-1], old_to_new_node_idx[old_node_idx], root_idxs)
#
#         log_q = proposal_dist.get_log_q(state)
#
#         particle = kernel.create_particle(data[data_idx], log_q, constrained_path[-1], state)
#
#         constrained_path.append(particle)
#
#     assert nx.is_isomorphic(graph, get_graph(constrained_path[-1], sigma))
#
#     return constrained_path


def get_data_to_node_map(tree):
    result = {}

    for node_idx in tree.data_points:
        for data_idx in tree.data_points[node_idx]:
            result[data_idx] = node_idx

    return result

# def get_data_to_node_map(graph):
#     result = {}
#
#     for node in graph.nodes_iter():
#         node_data = graph.node[node]
#
#         for x in node_data['data_points']:
#             result[x] = node
#
#     return result


def get_labels(graph):
    labels = {}

    for node in graph.nodes():
        for data_point in graph.node[node]['data_points']:
            labels[data_point] = node

    return [labels[x] for x in sorted(labels)]
