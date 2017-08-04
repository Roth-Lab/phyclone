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
from fscrp.math_utils import log_factorial, log_normalize


MarginalParticle = namedtuple('MarginalParticle', ['log_w', 'parent_particle', 'state'])


class State(object):

    def __init__(self, nodes, node_idx, log_p_prior, root_idxs):
        self.log_p_prior = log_p_prior

        assert node_idx < len(nodes)

        self.nodes = nodes

        self.node_idx = node_idx

        self._root_idxs = tuple(sorted(root_idxs))

        self._log_p = None

    def __key(self):
        return (self.node_idx, self._root_idxs)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def log_p(self):
        if self._log_p is None:
            self._log_p = MarginalNode(-1, self.root_nodes, self.nodes.values()[0].grid_size).log_p

        return self.log_p_prior + self._log_p

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

    def create_particle(self, data_point, log_q, parent_particle, state):
        '''
        Create a descendant particle from a parent particle
        '''
        if parent_particle is None:
            log_w = state.log_p - log_q

        else:
            log_w = state.log_p - parent_particle.state.log_p - log_q

        return MarginalParticle(log_w, parent_particle, state)

    def create_state(self, data_point, parent_particle, node_idx, root_idxs):
        log_p_prior = self._compute_log_p_prior(data_point, parent_particle, node_idx, root_idxs)

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

        return State(nodes, node_idx, log_p_prior, root_idxs)

    def propose_particle(self, data_point, parent_particle):
        '''
        Propose a particle for t given a particle from t - 1 and a data point.
        '''
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        state = proposal_dist.sample_state()

        log_q = proposal_dist.get_log_q(state)

        return self.create_particle(data_point, log_q, parent_particle, state)

    def _compute_log_p_prior(self, data_point, parent_particle, node_idx, root_idxs):
        if parent_particle is None:
            log_p = np.log(self.alpha)

        elif node_idx in parent_particle.state.root_idxs:
            node_counts = get_num_data_points_per_node(parent_particle)

            log_p = np.log(node_counts[node_idx])

        else:
            child_idxs = parent_particle.state.root_idxs - root_idxs

            node_counts = get_num_data_points_per_node(parent_particle)

            num_nodes = len(parent_particle.state.nodes)

            # CRP prior
            log_p = log(self.alpha)

            # Tree prior
            log_p += (num_nodes - 1) * log(num_nodes + 1) - num_nodes * log(num_nodes + 2)

            log_perm_norm = log_factorial(sum([node_counts[idx] for idx in child_idxs]))

            for idx in child_idxs:
                log_perm_norm -= log_factorial(node_counts[idx])

            log_p -= log_perm_norm

        return log_p


class BootstrapProposal(object):

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

    def get_log_q(self, state):
        if self.parent_particle is None:
            log_q = 0

        elif state.node_idx in self.parent_particle.state.root_idxs:
            num_roots = len(state.root_idxs)

            log_q = np.log(0.5) - np.log(num_roots)

        else:
            old_num_roots = len(self.parent_particle.state.root_idxs)

            log_q = np.log(0.5) - old_num_roots * np.log(2)

        return log_q

    def sample_state(self):
        if self.parent_particle is None:
            state = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            u = random.random()

            if u < 0.5:
                state = self._propose_existing_node()

            else:
                state = self._propose_new_node()

        return state

    def _propose_existing_node(self):
        node_idx = random.choice(list(self.parent_particle.state.root_idxs))

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            self.parent_particle.state.root_idxs
        )

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.state.root_idxs)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.state.root_idxs, num_children)

        node_idx = max(self.parent_particle.state.nodes.keys() + [-1, ]) + 1

        root_idxs = self.parent_particle.state.root_idxs - set(children)

        root_idxs.add(node_idx)

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            root_idxs
        )


class MarginalBootstrapKernel(MarginalKernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposal(data_point, self, parent_particle)


class FullyAdaptedProposal(object):

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self._init_dist()

    def get_log_q(self, state):
        return self.log_q[self.states.index(state)]

    def sample_state(self):
        q = np.exp(self.log_q)

        idx = np.random.multinomial(1, q).argmax()

        return self.states[idx]

    def _init_dist(self):
        self.states = self._propose_new_node()

        if self.parent_particle is not None:
            self.states.extend(self._propose_existing_node())

        log_q = [x.log_p for x in self.states]

        self.log_q = log_normalize(np.array(log_q))

    def _propose_existing_node(self):
        proposed_states = []

        for node_idx in self.parent_particle.state.root_idxs:
            proposed_states.append(
                self.kernel.create_state(
                    self.data_point,
                    self.parent_particle,
                    node_idx,
                    self.parent_particle.state.root_idxs
                )
            )

        return proposed_states

    def _propose_new_node(self):
        if self.parent_particle is None:
            return [
                self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))
            ]

        proposed_states = []

        node_idx = max(self.parent_particle.state.nodes.keys() + [-1, ]) + 1

        num_roots = len(self.parent_particle.state.root_idxs)

        for r in range(0, num_roots + 1):
            for child_idxs in itertools.combinations(self.parent_particle.state.root_idxs, r):
                root_idxs = self.parent_particle.state.root_idxs - set(child_idxs)

                root_idxs.add(node_idx)

                proposed_states.append(
                    self.kernel.create_state(self.data_point, self.parent_particle, node_idx, root_idxs)
                )

        return proposed_states


class MarginalFullyAdaptedKernel(MarginalKernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposal(data_point, self, parent_particle)


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

        proposal_dist = kernel.get_proposal_distribution(data[data_idx], constrained_path[-1])

        state = kernel.create_state(data[data_idx], constrained_path[-1], old_to_new_node_idx[old_node_idx], root_idxs)

        log_q = proposal_dist.get_log_q(state)

        particle = kernel.create_particle(data[data_idx], log_q, constrained_path[-1], state)

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
