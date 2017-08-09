'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import namedtuple

import numpy as np

from fscrp.kernels.marginal.data_structures import MarginalNode
from fscrp.kernels.marginal.utils import get_num_data_points_per_node
from fscrp.math_utils import log_factorial


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
            log_p = np.log(self.alpha)

            # Tree prior
            log_p += (num_nodes - 1) * np.log(num_nodes + 1) - num_nodes * np.log(num_nodes + 2)

            log_perm_norm = log_factorial(sum([node_counts[idx] for idx in child_idxs]))

            for idx in child_idxs:
                log_perm_norm -= log_factorial(node_counts[idx])

            log_p -= log_perm_norm

        return log_p
