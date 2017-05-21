'''
Created on 8 May 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict, namedtuple
from math import log

import itertools
import networkx as nx
import numpy as np
import random

from fscrp.data_structures import MarginalNode
from fscrp.particle_utils import iter_particles
from fscrp.math_utils import exp_normalize, log_sum_exp, discrete_rvs

MarginalParticle = namedtuple('MarginalParticle', ['log_w', 'parent_particle', 'state'])


class State(object):

    def __init__(self, nodes, node_idx, root_idxs):
        self.nodes = nodes

        self.node_idx = node_idx

        self._root_idxs = tuple(sorted(root_idxs))

    def __key(self):
        return (self.node_idx, self._root_idxs)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def root_idxs(self):
        return set(self._root_idxs)

    @property
    def root_nodes(self):
        return [self.nodes[idx] for idx in self._root_idxs]


class MarginalKernel(object):

    def __init__(self, alpha, grid_size):
        self.alpha = alpha

        self.grid_size = grid_size

    def create_particle(self, data_point, parent_particle, state, log_q=None, log_q_norm=None):
        '''
        Create a descendant particle from a parent particle
        '''
        if log_q is None:
            log_q = self.get_log_q(data_point, parent_particle)

        if log_q_norm is None:
            log_q_norm = log_sum_exp(np.array(log_q.values()))

        return MarginalParticle(log_q_norm, parent_particle, state)

    def create_state(self, data_point, parent_particle, node_idx, root_idxs):
        if parent_particle is None:
            assert node_idx == 0

            assert root_idxs == set([0, ])

            nodes = {}

            nodes[node_idx] = MarginalNode(node_idx, (), self.grid_size)

        elif node_idx in parent_particle.state.nodes:
            assert root_idxs == parent_particle.state.root_idxs

            nodes = parent_particle.state.nodes.copy()

            nodes[node_idx] = parent_particle.state.nodes[node_idx].copy()

        else:
            child_idxs = parent_particle.state.root_idxs - root_idxs

            nodes = parent_particle.state.nodes.copy()

            child_nodes = [parent_particle.state.nodes[idx] for idx in child_idxs]

            nodes[node_idx] = MarginalNode(node_idx, child_nodes, self.grid_size)

        nodes[node_idx].add_data_point(data_point)

        return State(nodes, node_idx, root_idxs)

    def get_log_q(self, data_point, parent_particle):
        log_q = self._get_log_q_new(data_point, parent_particle)

        if parent_particle is not None:
            log_q.update(self._get_log_q_existing(data_point, parent_particle))

        for state in log_q:
            log_q[state] += self._get_tree_log_p(state)

            if parent_particle is not None:
                log_q[state] -= self._get_tree_log_p(parent_particle.state)

        return log_q

    def propose_particle(self, data_point, parent_particle):
        '''
        Propose a particle for t given a particle from t - 1 and a data point.
        '''
        log_q = self.get_log_q(data_point, parent_particle)

        state_probs, log_q_norm = exp_normalize(np.array(log_q.values()))

        state_idx = discrete_rvs(state_probs)

        state = log_q.keys()[state_idx]

        return self.create_particle(data_point, parent_particle, state, log_q=log_q, log_q_norm=log_q_norm)

    def _get_log_q_existing(self, data_point, parent_particle):
        log_q = {}

        cluster_sizes = get_num_data_points_per_node(parent_particle)

        for node_idx in parent_particle.state.root_idxs:
            state = self.create_state(data_point, parent_particle, node_idx, parent_particle.state.root_idxs)

            # CRP prior
            log_q[state] = log(cluster_sizes[node_idx])

        return log_q

    def _get_log_q_new(self, data_point, parent_particle):
        log_q = {}

        if parent_particle is None:
            node_idx = 0

            root_idxs = set([node_idx, ])

            state = self.create_state(data_point, parent_particle, node_idx, root_idxs)

            log_q[state] = log(self.alpha)

        else:
            node_idx = max(parent_particle.state.nodes.keys() + [-1, ]) + 1

            num_nodes = len(parent_particle.state.nodes)

            num_roots = len(parent_particle.state.root_idxs)

            for r in range(0, num_roots + 1):
                for child_idxs in itertools.combinations(parent_particle.state.root_idxs, r):
                    root_idxs = parent_particle.state.root_idxs - set(child_idxs)

                    root_idxs.add(node_idx)

                    state = self.create_state(data_point, parent_particle, node_idx, root_idxs)

                    # CRP prior
                    log_q[state] = log(self.alpha)

                    # Tree prior
                    log_q[state] += (num_nodes - 1) * log(num_nodes + 1) - num_nodes * log(num_nodes + 2)

        return log_q

    def _get_tree_log_p(self, state):
        return MarginalNode(-1, state.root_nodes, self.grid_size).log_p


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
    )

    for idx in particle.state.root_idxs:
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
            node_data_points[particle.state.node_idx].append(i)

        else:
            node_data_points[particle.state.node_idx].append(sigma[i])

    return node_data_points


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

    old_to_new_node_idx = {}

    root_idxs = set()

    for data_idx in sigma:
        old_node_idx = data_to_node[data_idx]

        if old_node_idx not in old_to_new_node_idx:
            for child_idx in graph.successors(old_node_idx):
                root_idxs.remove(old_to_new_node_idx[child_idx])

            old_to_new_node_idx[old_node_idx] = node_idx

            root_idxs.add(node_idx)

            node_idx += 1

        state = kernel.create_state(data[data_idx], constrained_path[-1], old_to_new_node_idx[old_node_idx], root_idxs)

        particle = kernel.create_particle(data[data_idx], constrained_path[-1], state)

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
