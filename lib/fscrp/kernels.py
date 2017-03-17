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

    def __init__(self, alpha, log_likelihood_func, node_param_proposal, node_agg_func, node_param_prior_func):
        self.alpha = alpha

        self.log_likelihood_func = log_likelihood_func

        self.node_param_proposal = node_param_proposal

        self.node_agg_func = node_agg_func

        self.node_param_prior_func = node_param_prior_func

    def propose_particle(self, data_point, parent_particle, seed=None):
        if seed is not None:
            old_state = random.getstate()

            random.seed(seed)

        node = self.propose_node(data_point, parent_particle)

        log_p = self.log_likelihood_func(data_point, node.agg_params)

        log_q = self.get_log_q(data_point, node, parent_particle)

        log_w = self._compute_log_weight(node, parent_particle, log_q, log_p)

        if seed is not None:
            random.setstate(old_state)

        return Particle(log_w=log_w, node=node, parent_particle=parent_particle)

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

            log_weight += self.node_param_prior_func(node, nodes)

        elif num_data_points > 1:
            log_weight += log(num_data_points - 1)

        # Uniform prior over trees
        if num_partitions > 1:
            log_weight += (num_partitions - 2) * log(num_partitions - 1) - (num_partitions - 1) * log(num_partitions)

        return log_weight


class BootstrapKernel(Kernel):

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
        nodes = get_nodes(particle)

        node = random.sample(nodes, 1)[0]

        return node

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
        nodes = get_nodes(parent_particle)

        return -log(len(nodes))

    def _get_log_q_add_new(self, data_point, node, parent_particle):
        log_q = self.node_param_proposal.log_p(node.node_params, data_point, get_node_params(parent_particle))

        root_nodes = get_root_nodes(parent_particle)

        num_root = len(root_nodes)

        log_q -= num_root * log(2)

        return log_q


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

        node_idx = discrete_rvs(p)

        node = nodes[node_idx]

        return node

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

        cluster_sizes = get_num_data_points_per_node(parent_particle)

        log_p = []

        for node in nodes:
            log_p.append(log(cluster_sizes[node]) + self.log_likelihood_func(data_point, node.agg_params))

        node_idx = nodes.index(node)

        node = nodes[node_idx]

        log_q = log_p[node_idx] - log_sum_exp(log_p)

        return log_q

    def _get_log_q_add_new(self, data_point, node, parent_particle):
        log_q = self.node_param_proposal.log_p(node.node_params, data_point, get_node_params(parent_particle))

        root_nodes = get_root_nodes(parent_particle)

        num_root = len(root_nodes)

        log_q -= num_root * log(2)

        return log_q
