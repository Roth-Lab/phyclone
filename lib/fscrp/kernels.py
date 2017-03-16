'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from math import log
from pydp.rvs import discrete_rvs

import random

from fscrp.data_structures import Node, Particle
from fscrp.particle_utils import get_nodes, get_num_data_points_per_node, get_node_params, get_root_nodes
from fscrp.utils import exp_normalize
from pydp.utils import log_sum_exp


class Kernel(object):

    def __init__(self, alpha, log_likelihood_func, node_param_proposal_func, node_agg_func, node_param_prior_func):
        self.alpha = alpha

        self._log_likelihood_func = log_likelihood_func

        self._node_param_proposal_func = node_param_proposal_func

        self._node_agg_func = node_agg_func

        self._node_param_prior_func = node_param_prior_func

    def propose_particle(self, data_point, parent_particle, seed=None):
        if seed is not None:
            old_state = random.getstate()

            random.seed(seed)

        log_q, node = self._propose_node(data_point, parent_particle)

        log_p = self._log_likelihood_func(data_point, node.agg_params)

        log_w = self._compute_log_weight(node, parent_particle, log_q, log_p)

        if seed is not None:
            random.setstate(old_state)

        return Particle(data_point=data_point, log_w=log_w, node=node, parent_particle=parent_particle)

    def _compute_log_weight(self, node, parent_particle, log_q, log_p):
        # Likelihood prior
        log_weight = log_p - log_q

        nodes = get_nodes(parent_particle)

        partition_sizes = get_num_data_points_per_node(parent_particle)

        # Add current node
        partition_sizes[node] += 1

        num_partitions = len(nodes)

        num_data_points = partition_sizes[node]

        # CRP prior
        if num_data_points == 1:
            log_weight += log(self.alpha)

            num_partitions += 1

            if num_partitions > 1:
                log_weight += log(num_partitions - 1)

            nodes.add(node)

            log_weight += self._node_param_prior_func(node, nodes)

        elif num_data_points > 1:
            log_weight += log(num_data_points - 1)

        # Uniform prior over trees
        if num_partitions > 1:
            log_weight += (num_partitions - 2) * log(num_partitions - 1) - (num_partitions - 1) * log(num_partitions)

        return log_weight


class BootstrapKernel(Kernel):

    def _propose_node(self, data_point, parent_particle):
        nodes = get_nodes(parent_particle)

        if len(nodes) == 0:
            proposal_funcs = [self._add_data_point_to_new_node, ]

        else:
            proposal_funcs = [self._add_data_point_to_existing_node, self._add_data_point_to_new_node]

        proposal_func = random.choice(proposal_funcs)

        log_q, node = proposal_func(data_point, parent_particle)

        return -log(len(proposal_funcs)) + log_q, node

    def _add_data_point_to_existing_node(self, data_point, particle):
        nodes = get_nodes(particle)

        node = random.sample(nodes, 1)[0]

        return -log(len(nodes)), node

    def _add_data_point_to_new_node(self, data_point, particle):
        root_nodes = get_root_nodes(particle)

        num_root = len(root_nodes)

        num_children = random.randint(0, num_root)

        if num_children > 0:
            children = random.sample(root_nodes, num_children)

        else:
            children = []

        node_params, log_q = self._node_param_proposal_func(data_point, get_node_params(particle))

        subtree_params = [node_params, ]

        for child in children:
            subtree_params.append(child.agg_params)

        agg_params = self._node_agg_func(subtree_params)

        node = Node(tuple(children), node_params, agg_params)

        log_q -= num_root * log(2)

        return log_q, node


class SemiAdaptedKernel(Kernel):

    def _propose_node(self, data_point, parent_particle):
        nodes = get_nodes(parent_particle)

        if len(nodes) == 0:
            proposal_funcs = [self._add_data_point_to_new_node, ]

        else:
            proposal_funcs = [self._add_data_point_to_existing_node, self._add_data_point_to_new_node]

        proposal_func = random.choice(proposal_funcs)

        log_q, node = proposal_func(data_point, parent_particle)

        return -log(len(proposal_funcs)) + log_q, node

    def _add_data_point_to_existing_node(self, data_point, particle):
        nodes = list(get_nodes(particle))

        cluster_sizes = get_num_data_points_per_node(particle)

        log_p = []

        for node in nodes:
            log_p.append(log(cluster_sizes[node]) + self._log_likelihood_func(data_point, node.agg_params))

        p = exp_normalize(log_p)

        node_idx = discrete_rvs(p)

        node = nodes[node_idx]

        log_q = log_p[node_idx] - log_sum_exp(log_p)

        return log_q, node

    def _add_data_point_to_new_node(self, data_point, particle):
        root_nodes = get_root_nodes(particle)

        num_root = len(root_nodes)

        num_children = random.randint(0, num_root)

        if num_children > 0:
            children = random.sample(root_nodes, num_children)

        else:
            children = []

        node_params, log_q = self._node_param_proposal_func(data_point, get_node_params(particle))

        subtree_params = [node_params, ]

        for child in children:
            subtree_params.append(child.agg_params)

        agg_params = self._node_agg_func(subtree_params)

        node = Node(tuple(children), node_params, agg_params)

        log_q -= num_root * log(2)

        return log_q, node
