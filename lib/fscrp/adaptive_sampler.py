'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

from math import log

import numpy as np
import random

from fscrp.data_structures import Node, Particle
from fscrp.particle_utils import get_nodes, get_node_params, get_num_data_points_per_node, get_root_nodes
from fscrp.swarm import ParticleSwarm


class AdaptiveSampler(object):

    def __init__(
            self,
            alpha,
            data_points,
            num_particles,
            log_likelihood_func,
            node_param_proposal_func,
            node_agg_func,
            node_param_prior_func,
            resample_threshold=0.5):

        self.alpha = alpha

        self.data_points = data_points

        self.iteration = 0

        self.num_iterations = len(data_points)

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self._log_likelihood_func = log_likelihood_func

        self._node_param_proposal_func = node_param_proposal_func

        self._node_agg_func = node_agg_func

        self._node_param_prior_func = node_param_prior_func

    def sample(self):
        self._init_swarm()

        for _ in range(self.num_iterations):
            print 'Iteration {0} of {1}.'.format(self.iteration, self.num_iterations)

            self._sample_new_particles()

            self._resample_if_necessary()

            self.iteration += 1

        return self.swarm.to_dict()

    def _init_swarm(self):
        self.swarm = ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        for _ in range(self.num_particles):
            self.swarm.add_particle(uniform_weight, None)

    def _resample_if_necessary(self):
        swarm = self.swarm

        if swarm.relative_ess <= self.resample_threshold:
            new_swarm = ParticleSwarm()

            print 'Resampling', swarm.relative_ess

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = np.random.multinomial(self.num_particles, swarm.weights)

            for particle, multiplicity in zip(swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    new_swarm.add_particle(log_uniform_weight, particle)

        else:
            new_swarm = swarm

        self.swarm = new_swarm

    def _sample_new_particles(self):
        new_swarm = ParticleSwarm()

        for parent_log_W, parent_particle in zip(self.swarm.log_weights, self.swarm.particles):
            particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + particle.log_w, particle)

        self.swarm = new_swarm

    def _propose_particle(self, parent_particle):
        data_point = self.data_points[self.iteration]

        log_q, node = self._propose_node(data_point, parent_particle)

        log_p = self._log_likelihood_func(data_point, node.agg_params)

        log_w = self._compute_log_weight(node, parent_particle, log_q, log_p)

        return Particle(data_point=data_point, log_w=log_w, node=node, parent_particle=parent_particle)

    def _propose_node(self, data_point, parent_particle):
        nodes = get_nodes(parent_particle)

        if len(nodes) == 0:
            proposal_funcs = [self._add_data_point_to_new_node, ]

        else:
            proposal_funcs = [self._add_data_point_to_existing_node, self._add_data_point_to_new_node]

        proposal_func = random.choice(proposal_funcs)

        log_q, node = proposal_func(data_point, parent_particle)

        return -log(len(proposal_funcs)) + log_q, node

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

        log_weight -= log(self.alpha + self.iteration + 1)

        # Uniform prior over trees
        if num_partitions > 1:
            log_weight += (num_partitions - 2) * log(num_partitions - 1) - (num_partitions - 1) * log(num_partitions)

        return log_weight

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
