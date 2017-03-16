'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

from math import log
from pydp.rvs import multinomial_rvs

import random

from fscrp.data_structures import Node, Particle
from fscrp.particle_utils import get_data_points, get_nodes, get_node_params, get_num_data_points_per_node, \
    get_root_nodes
from fscrp.utils import exp_normalize


class ImplicitSampler(object):

    def __init__(
            self,
            alpha,
            data_points,
            num_particles,
            log_likelihood_func,
            node_param_proposal_func,
            node_agg_func,
            node_param_prior_func,
            num_implicit_particles=None):

        self.alpha = alpha

        self.data_points = data_points

        self.iteration = 0

        self.num_implicit_particles = num_implicit_particles

        self.num_iterations = len(data_points)

        self.num_particles = num_particles

        self.particle_multiplicities = {}

        self._log_likelihood_func = log_likelihood_func

        self._node_param_proposal_func = node_param_proposal_func

        self._node_agg_func = node_agg_func

        self._node_param_prior_func = node_param_prior_func

    def sample(self):
        self._init_swarm()

        for _ in range(self.num_iterations):
            print 'Iteration {0} of {1}.'.format(self.iteration, self.num_iterations)

            print '{0} distinct particles.'.format(len(self.particle_multiplicities))

            self._sample_new_particles()

            self.iteration += 1

        print len(get_data_points(self.particle_multiplicities.keys()[0]))

        return self.particle_multiplicities

    def _init_swarm(self):
        self.particle_multiplicities[None] = self.num_particles

    def _sample_new_particles(self):
        if self.num_implicit_particles is not None:
            self._expand_particles()

        log_weights = self._compute_log_weights()

        seeds = log_weights.keys()

        particle_probs = exp_normalize(log_weights.values())

        new_values = multinomial_rvs(self.num_particles, particle_probs)

        self.particle_multiplicities = {}

        for multiplicity, (particle, seed) in zip(new_values, seeds):
            if multiplicity == 0:
                continue

            new_particle = self._propose_particle(particle, seed)

            self.particle_multiplicities[new_particle] = multiplicity

    def _expand_particles(self):
        num_particles = sum(self.particle_multiplicities.values())

        probs = [x / num_particles for x in self.particle_multiplicities.values()]

        new_multiplicities = multinomial_rvs(self.num_implicit_particles, probs)

        self.particle_multiplicities = dict(zip(self.particle_multiplicities.keys(), new_multiplicities))

    def _compute_log_weights(self):
        log_weights = {}

        for particle in self.particle_multiplicities:
            for _ in range(self.particle_multiplicities[particle]):
                seed = random.randint(0, 1e15)

                new_particle = self._propose_particle(particle, seed)

                log_weights[(particle, seed)] = new_particle.log_w

        return log_weights

    def _propose_particle(self, parent_particle, seed):
        old_state = random.getstate()

        random.seed(seed)

        data_point = self.data_points[self.iteration]

        log_q, node = self._propose_node(data_point, parent_particle)

        log_p = self._log_likelihood_func(data_point, node.agg_params)

        log_w = self._compute_log_weight(node, parent_particle, log_q, log_p)

        random.setstate(old_state)

        return Particle(
            data_point=data_point,
            log_w=log_w,
            node=node,
            parent_particle=parent_particle
        )

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
