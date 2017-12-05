'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from fscrp.samplers.swarm import ParticleSwarm


class AdaptiveSampler(object):
    """ Unconditional SMC sampler
    """

    def __init__(self, data_points, kernel, num_particles, resample_threshold=0.5):
        self.data_points = data_points

        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self.iteration = 0

        self.num_iterations = len(data_points)

    def sample(self):
        self.log_Z = 0

        self._init_swarm()

        for _ in range(self.num_iterations - 1):
            self._sample_new_particles()

            self._resample_if_necessary()

            self.iteration += 1

        self._sample_new_particles()

        self.iteration += 1

        self.log_Z += -np.log(self.swarm.num_particles) + self.swarm.log_norm_const

        return self.swarm.to_dict()

    def _init_swarm(self):
        self.swarm = ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        for _ in range(self.num_particles):
            self.swarm.add_particle(uniform_weight, None)

    def _propose_particle(self, parent_particle):
        data_point = self.data_points[self.iteration]

        return self.kernel.propose_particle(data_point, parent_particle)

    def _resample_if_necessary(self):
        swarm = self.swarm

        if swarm.relative_ess <= self.resample_threshold:
            self.log_Z += -np.log(self.swarm.num_particles) + self.swarm.log_norm_const

            new_swarm = ParticleSwarm()

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
