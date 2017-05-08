'''
Created on 8 May 2017

@author: Andrew Roth
'''
from math import log

import random

from fscrp.data_structures import Particle
from fscrp.particle_utils import get_nodes, get_num_data_points_per_node


class Kernel(object):

    def __init__(self, alpha, log_likelihood_func, node_param_proposal, node_agg_func, node_param_prior_func):
        self.alpha = alpha

        self.log_likelihood_func = log_likelihood_func

        self.node_param_proposal = node_param_proposal

        self.node_agg_func = node_agg_func

        self.node_param_prior_func = node_param_prior_func

    def create_particle(self, data_point, node, parent_particle):
        log_w = self._compute_log_weight(data_point, node, parent_particle)

        return Particle(log_w=log_w, node=node, parent_particle=parent_particle)

    def propose_particle(self, data_point, parent_particle, seed=None):
        if seed is not None:
            old_state = random.getstate()

            random.seed(seed)

        node = self.propose_node(data_point, parent_particle)

        particle = self.create_particle(data_point, node, parent_particle)

        if seed is not None:
            random.setstate(old_state)

        return particle

    def _compute_log_weight(self, data_point, node, parent_particle):
        log_p = self.log_likelihood_func(data_point, node.agg_params)

        log_q = self.get_log_q(data_point, node, parent_particle)

        # Likelihood prior
        log_weight = log_p - log_q

        nodes = get_nodes(parent_particle)

        cluster_sizes = get_num_data_points_per_node(parent_particle)

        # Add current node
        num_partitions = len(nodes)

        # CRP prior
        if cluster_sizes[node] == 0:
            log_weight += log(self.alpha)

#             num_partitions += 1

            if num_partitions > 1:
                log_weight += (num_partitions - 1) * log(num_partitions) - num_partitions * log(num_partitions + 1)

            log_weight += self.node_param_prior_func(node, nodes)

#             log_weight -= (len(get_root_nodes(parent_particle)) - len(node.children) + 1)

        elif cluster_sizes[node] > 0:
            log_weight += log(cluster_sizes[node])

#         # Uniform prior over trees
#         if num_partitions > 1:
#             log_weight += (num_partitions - 2) * log(num_partitions - 1) - (num_partitions - 1) * log(num_partitions)

        return log_weight
