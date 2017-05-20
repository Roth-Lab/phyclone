'''
Created on 8 May 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict
from math import log
from pydp.rvs import discrete_rvs

import networkx as nx
import random

from fscrp.data_structures import MarginalNode, MarginalParticle
from fscrp.kernels.base import Kernel
from fscrp.particle_utils import iter_particles
from fscrp.utils import exp_normalize
from pydp.utils import log_sum_exp


class MarginalKernel(Kernel):

    def __init__(self, alpha, grid_size):
        self.alpha = alpha

        self.grid_size = grid_size

    def create_particle(self, data_point, node_idx, parent_particle, root_idxs):
        new_nodes = {}

        if parent_particle is None:
            parent_nodes = {}

            parent_root_idxs = set()

        else:
            parent_nodes = parent_particle.nodes

            parent_root_idxs = parent_particle.root_idxs.copy()

        for idx in parent_nodes:
            node = parent_particle.nodes[idx]

            if node_idx == idx:
                node = node.copy()

            new_nodes[idx] = node

        if root_idxs == parent_root_idxs:
            cluster_sizes = get_num_data_points_per_node(parent_particle)

            log_w = log(cluster_sizes[node_idx]) - log(len(root_idxs))

        else:
            children_idxs = parent_root_idxs - root_idxs

            children = [new_nodes[idx] for idx in children_idxs]

            new_nodes[node_idx] = MarginalNode(node_idx, children, self.grid_size)

            num_clusters = len(parent_nodes)

            log_w = 0

            if num_clusters > 1:
                log_w += (num_clusters - 1) * log(num_clusters + 1) - num_clusters * log(num_clusters + 2)

            log_w += log(self.alpha)

            log_w -= len(parent_root_idxs) * log(2)

        new_nodes[node_idx].add_data_point(data_point)

        log_w += MarginalNode(
            -1,
            [new_nodes[n] for n in root_idxs],
            self.grid_size
        ).log_p

        if parent_particle is not None:
            log_w -= MarginalNode(
                -1,
                [parent_particle.nodes[n] for n in parent_particle.root_idxs],
                self.grid_size
            ).log_p

        return MarginalParticle(log_w, parent_particle, new_nodes, node_idx, root_idxs)

    def propose_node(self, data_point, parent_particle):
        proposal_funcs = [self._add_data_point_to_new_node, ]

        if parent_particle is not None:
            proposal_funcs.append(self._add_data_point_to_existing_node)

        f = random.choice(proposal_funcs)

        return f(data_point, parent_particle)

    def propose_particle(self, data_point, parent_particle):
        node_idx, root_idxs = self.propose_node(data_point, parent_particle)

        particle = self.create_particle(data_point, node_idx, parent_particle, root_idxs)

        return particle

    def _add_data_point_to_existing_node(self, data_point, parent_particle):
        root_idxs = parent_particle.root_idxs.copy()

        node_idx = random.choice(list(root_idxs))

        return node_idx, root_idxs

    def _add_data_point_to_new_node(self, data_point, parent_particle):
        if parent_particle is None:
            node_idx = 0

            root_idxs = set([node_idx, ])

        else:
            root_idxs = parent_particle.root_idxs.copy()

            num_root = len(root_idxs)

            num_children = random.randint(0, num_root)

            children = random.sample(list(root_idxs), num_children)

            root_idxs = root_idxs - set(children)

            node_idx = max(parent_particle.nodes.keys() + [-1, ]) + 1

            root_idxs.add(node_idx)

        return node_idx, root_idxs

    def _get_tree_log_p(self, root_nodes):
        return MarginalNode(-1, root_nodes, self.grid_size).log_p


def update_nodes(node, parent_particle):
    graph = get_graph(parent_particle)

    parent = graph.predecessors(node.idx)

    while len(parent) > 0:
        assert len(parent) == 1

        graph.node[parent[0]]['node'].update()

        parent = graph.predecessors(graph.node[parent[0]]['node'].idx)


def get_num_data_points_per_node(last_particle):
    counts = defaultdict(int)

    for particle in iter_particles(last_particle):
        counts[particle.node_idx] += 1

    return dict(counts)


def get_root_nodes(last_particle):
    if last_particle is None:
        root_nodes = {}

    else:
        root_nodes = dict(zip([n.idx for n in last_particle.node.children], last_particle.node.children))

    return root_nodes


def get_nodes(last_particle):
    return last_particle.nodes


def get_graph(particle, sigma=None):
    graph = nx.DiGraph()

    nodes = get_nodes(particle)

    node_data_points = get_node_data_points(particle, sigma=sigma)

    graph.add_node(
        -1,
        data_points=[],
    )

    for idx in particle.root_idxs:
        graph.add_edge(-1, idx)

    for node_idx in node_data_points:
        graph.add_node(
            node_idx,
            data_points=node_data_points[node_idx],
        )

    for node in nodes.values():
        for child in node.children:
            graph.add_edge(node.idx, child.idx)

    return graph


def get_node_data_points(last_particle, sigma=None):
    node_data_points = defaultdict(list)

    for i, particle in enumerate(reversed(list(iter_particles(last_particle)))):
        if sigma is None:
            node_data_points[particle.node_idx].append(i)

        else:
            node_data_points[particle.node_idx].append(sigma[i])

    return node_data_points


def get_num_clusters(particle):
    return len(particle.nodes)


def sample_sigma(graph, source=None):
    if source is None:
        sigma = []

        for node in graph.successors(-1):
            sigma.append(sample_sigma(graph, source=node))

        return interleave_lists(sigma)

    child_sigma = []

    for child in graph.successors(source):
        child_sigma.append(sample_sigma(graph, source=child))

    sigma = interleave_lists(child_sigma)

    source_sigma = list(graph.node[source]['data_points'])

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


def get_constrained_path(data, graph, kernel, sigma):
    constrained_path = [None, ]

    data_to_node = get_data_to_node_map(graph)

    node_idx = 0

    root_idxs = set()

    visited_nodes = set()

    new_nodes = {}

    for data_idx in sigma:
        old_node_idx = data_to_node[data_idx]

        if old_node_idx not in visited_nodes:
            children = []

            for child_idx in graph.successors(old_node_idx):
                children.append(new_nodes[child_idx])

                root_idxs.remove(new_nodes[child_idx].idx)

            new_nodes[old_node_idx] = MarginalNode(node_idx, tuple(children), kernel.grid_size)

            root_idxs.add(node_idx)

            visited_nodes.add(old_node_idx)

            node_idx += 1

        particle = kernel.create_particle(
            data[data_idx],
            new_nodes[old_node_idx].idx,
            constrained_path[-1],
            root_idxs.copy()
        )

        constrained_path.append(particle)

    assert nx.is_isomorphic(graph, get_graph(constrained_path[-1], sigma))

    return constrained_path


def get_data_to_node_map(graph):
    result = {}

    for node in graph.nodes_iter():
        node_data = graph.node[node]

        for x in node_data['data_points']:
            result[x] = node

    return result
