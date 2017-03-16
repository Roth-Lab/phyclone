'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

from pydp.rvs import multinomial_rvs

import random

from fscrp.particle_utils import get_data_points
from fscrp.utils import exp_normalize


class ImplicitSampler(object):

    def __init__(
            self,
            data_points,
            kernel,
            num_particles,
            num_implicit_particles=None):

        self.data_points = data_points

        self.kernel = kernel

        self.num_particles = num_particles

        self.num_implicit_particles = num_implicit_particles

        self.iteration = 0

        self.num_iterations = len(data_points)

        self.particle_multiplicities = {}

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
        data_point = self.data_points[self.iteration]

        return self.kernel.propose_particle(data_point, parent_particle, seed=seed)
