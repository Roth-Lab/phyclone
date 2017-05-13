'''
Created on 8 May 2017

@author: Andrew Roth
'''
from math import log
from pydp.rvs import discrete_rvs

import random

from fscrp.data_structures import MarginalNode, MarginalParticle
from fscrp.kernels.base import Kernel
from fscrp.particle_utils import get_nodes, get_num_data_points_per_node, get_root_nodes
from fscrp.utils import exp_normalize
from pydp.utils import log_sum_exp


class MarginalKernel(Kernel):

    def __init__(self, alpha, grid_size):
        self.alpha = alpha

        self.grid_size = grid_size

    def create_particle(self, data_point, node, parent_particle):
        log_q = self._get_log_q(data_point, parent_particle)

        log_w = log_sum_exp(log_q.values())

        new_root_nodes = list(get_root_nodes(parent_particle))

        [new_root_nodes.remove(x) for x in node.children if x in new_root_nodes]

        new_root_nodes.append(node)

        tree_log_p = self._get_tree_log_p(new_root_nodes)

        return MarginalParticle(log_w=log_w, node=node, parent_particle=parent_particle, tree_log_p=tree_log_p)

    def propose_node(self, data_point, parent_particle):
        log_q = self._get_log_q(data_point, parent_particle)

        node_probs = exp_normalize(log_q.values())

        node_idx = discrete_rvs(node_probs)

        node = log_q.keys()[node_idx]

        node.add_data_point(data_point)

        return node

    def _get_log_q(self, data_point, parent_particle):
        log_q = {}

        log_q.update(self._add_data_point_to_new_node(data_point, parent_particle))

        if parent_particle is not None:
            log_q.update(self._add_data_point_to_existing_node(data_point, parent_particle))

        return log_q

    def _add_data_point_to_existing_node(self, data_point, parent_particle):
        root_nodes = list(get_root_nodes(parent_particle))

        cluster_sizes = get_num_data_points_per_node(parent_particle)

        log_q = {}

        for node in root_nodes:
            node.add_data_point(data_point)

            tree_log_p = self._get_tree_log_p(root_nodes)

            log_q[node] = log(cluster_sizes[node]) + tree_log_p - parent_particle.tree_log_p

            node.remove_data_point(data_point)

        return log_q

    def _add_data_point_to_new_node(self, data_point, parent_particle):
        root_nodes = get_root_nodes(parent_particle)

        num_root = len(root_nodes)

        nodes = get_nodes(parent_particle)

        num_clusters = len(nodes)

        if num_clusters > 1:
            log_p_forest = (num_clusters - 1) * log(num_clusters + 1) - num_clusters * log(num_clusters + 2)

        else:
            log_p_forest = 0

        log_p_partition = log(self.alpha)

        if parent_particle is None:
            old_tree_log_p = 0

        else:
            old_tree_log_p = parent_particle.tree_log_p

        log_q = {}

        for num_children in range(0, num_root + 1):
            #             for children in itertools.combinations(root_nodes, num_children):
            children = random.sample(root_nodes, num_children)

            node = MarginalNode(children, self.grid_size)

            node.add_data_point(data_point)

            new_root_nodes = set(root_nodes) - set(children)

            new_root_nodes.add(node)

            new_root_nodes = tuple(new_root_nodes)

            tree_log_p = self._get_tree_log_p(new_root_nodes)

            log_q[node] = log_p_partition + log_p_forest + tree_log_p - old_tree_log_p

            node.remove_data_point(data_point)

        return log_q

    def _get_tree_log_p(self, root_nodes):
        return MarginalNode(root_nodes, self.grid_size).log_p
