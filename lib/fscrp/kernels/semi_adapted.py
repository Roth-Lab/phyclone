'''
Created on 8 May 2017

@author: Andrew Roth
'''
from __future__ import division

from math import log

import random
import numpy as np

from fscrp.data_structures import Node
from fscrp.kernels.base import Kernel
from fscrp.particle_utils import get_nodes, get_num_data_points_per_node, get_node_params, get_root_nodes
from fscrp.utils import exp_normalize
from pydp.utils import log_sum_exp


class SemiAdaptedKernel(Kernel):

    def get_log_q(self, data_point, node, parent_particle):
        nodes = get_nodes(parent_particle)

        if node in nodes:
            log_q = self._get_log_q_add_existing(data_point, node, parent_particle)

        else:
            log_q = self._get_log_q_add_new(data_point, node, parent_particle)

        if parent_particle is not None:
            log_q -= log(2)

        return log_q

    def propose_node(self, data_point, parent_particle):
        if parent_particle is None:
            proposal_funcs = [self._add_data_point_to_new_node, ]

        else:
            proposal_funcs = [self._add_data_point_to_existing_node, self._add_data_point_to_new_node]

        proposal_func = random.choice(proposal_funcs)

        node = proposal_func(data_point, parent_particle)

        return node

    def _add_data_point_to_existing_node(self, data_point, particle):
        nodes = list(get_nodes(particle))

        cluster_sizes = get_num_data_points_per_node(particle)

        log_p = []

        for node in nodes:
            log_p.append(log(cluster_sizes[node]) + self.log_likelihood_func(data_point, node.agg_params))

        p = exp_normalize(log_p)

        node_idx = np.random.multinomial(1, p).argmax()

        return nodes[node_idx]

    def _add_data_point_to_new_node(self, data_point, particle):
        root_nodes = get_root_nodes(particle)

        num_root = len(root_nodes)

        num_children = random.randint(0, num_root)

        if num_children > 0:
            children = random.sample(root_nodes, num_children)

        else:
            children = []

        node_params = self.node_param_proposal.propose(data_point, get_node_params(particle))

        subtree_params = [node_params, ]

        for child in children:
            subtree_params.append(child.agg_params)

        agg_params = self.node_agg_func(subtree_params)

        node = Node(tuple(children), node_params, agg_params)

        return node

    def _get_log_q_add_existing(self, data_point, node, parent_particle):
        nodes = list(get_nodes(parent_particle))

        node_idx = nodes.index(node)

        cluster_sizes = get_num_data_points_per_node(parent_particle)

        log_p = []

        for test_node in nodes:
            log_p.append(log(cluster_sizes[test_node]) + self.log_likelihood_func(data_point, test_node.agg_params))

        log_q = log_p[node_idx] - log_sum_exp(log_p)

        return log_q

    def _get_log_q_add_new(self, data_point, node, parent_particle):
        log_q = self.node_param_proposal.log_likelihood(node.node_params, data_point, get_node_params(parent_particle))

        root_nodes = get_root_nodes(parent_particle)

        num_root = len(root_nodes)

        log_q -= num_root * log(2)

        return log_q
