'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from collections import defaultdict

import networkx as nx


def get_data_points(last_particle):
    data_points = []

    for particle in iter_particles(last_particle):
        data_points.append(particle.data_point)

    data_points.reverse()

    return data_points


def get_log_weights(last_particle):
    log_weights = []

    for particle in iter_particles(last_particle):
        log_weights.append(particle.log_w)

    log_weights.reverse()

    return log_weights


def get_log_likelihood(last_particle):
    log_likelihood = 0

    for particle in iter_particles(last_particle):
        log_likelihood += particle.log_likelihood

    return log_likelihood


def get_nodes(last_particle):
    nodes = set()

    for particle in iter_particles(last_particle):
        nodes.add(particle.node)

    return nodes


def get_node_data_points(last_particle):
    node_data_points = defaultdict(list)

    for particle in iter_particles(last_particle):
        node_data_points[particle.node].append(particle.data_point)

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

    for node in get_nodes(last_particle):
        child_nodes.update(set(node.children))

    root_nodes = nodes - child_nodes

    return root_nodes


def iter_particles(particle):
    while particle is not None:
        yield particle

        particle = particle.parent_particle


def get_graph(last_particle, multiplicity):
    log_weights = get_log_weights(last_particle)

    data_points = get_data_points(last_particle)

    graph = nx.DiGraph(log_weights=log_weights,
                       data_points=[x.id for x in data_points],
                       multiplicity=multiplicity)

    nodes = get_nodes(last_particle)

    node_data_points = get_node_data_points(last_particle)

    node_ids = {}

    for i, node in enumerate(nodes):
        node_ids[node] = "Node {0}".format(i + 1)

        graph.add_node(node_ids[node],
                       agg_params=node.agg_params,
                       node_params=node.node_params,
                       data_points=[x.id for x in node_data_points[node]])

    for i, node in enumerate(nodes):
        for child in node.children:
            graph.add_edge(node_ids[node], node_ids[child])

    return graph
