'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict

import networkx as nx


def get_constrained_path(data, graph, kernel, sigma):
    constrained_path = [None, ]

    data_to_node = get_data_to_node_map(graph)

    for idx in sigma:
        node = data_to_node[idx]

        particle = kernel.create_particle(data[idx], node, constrained_path[-1])

        constrained_path.append(particle)

    return constrained_path


def get_data_to_node_map(graph):
    result = {}

    for node in graph.nodes_iter():
        node_data = graph.node[node]

        for x in node_data['data_points']:
            result[x] = node_data['node']

    return result


def get_data_points(last_particle):
    data_points = []

    for particle in iter_particles(last_particle):
        data_points.append(particle.data_point)

    data_points.reverse()

    return data_points


def get_log_likelihood(last_particle):
    log_likelihood = 0

    for particle in iter_particles(last_particle):
        log_likelihood += particle.log_likelihood

    return log_likelihood


def get_log_weights(last_particle):
    log_weights = []

    for particle in iter_particles(last_particle):
        log_weights.append(particle.log_w)

    log_weights.reverse()

    return log_weights


def get_nodes(last_particle):
    nodes = set()

    for particle in iter_particles(last_particle):
        nodes.add(particle.node)

    return nodes


def get_node_data_points(last_particle, sigma=None):
    node_data_points = defaultdict(list)

    for i, particle in enumerate(reversed(list(iter_particles(last_particle)))):
        node = particle.node

        if sigma is None:
            node_data_points[node].append(i)

        else:
            node_data_points[node].append(sigma[i])

    return node_data_points


def get_node_params(last_particle):
    node_params = []

    for node in get_nodes(last_particle):
        node_params.append(node.node_params)

    return node_params


def get_num_data_points_per_node(last_particle):
    num_data_points = defaultdict(int)

    for node in get_nodes(last_particle):
        num_data_points[node] += 1

    return num_data_points


def get_root_nodes(last_particle):
    child_nodes = set()

    nodes = get_nodes(last_particle)

    for node in nodes:
        child_nodes.update(set(node.children))

    root_nodes = nodes - child_nodes

    return root_nodes


def iter_particles(particle):
    while particle is not None:
        yield particle

        particle = particle.parent_particle


def get_graph(particle, sigma=None):
    graph = nx.DiGraph()

    nodes = get_nodes(particle)

    node_data_points = get_node_data_points(particle, sigma=sigma)

    node_map = {}

    for node_id, node in enumerate(nodes):
        node_map[node] = node_id

        graph.add_node(
            node_id,
            data_points=node_data_points[node],
            node=node,
        )

    for node in nodes:
        for child in node.children:
            graph.add_edge(node_map[node], node_map[child])

    return graph
