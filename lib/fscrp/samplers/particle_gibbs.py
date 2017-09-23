'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from fscrp.samplers.swarm import ParticleSwarm


class ParticleGibbsSampler(object):

    def __init__(
            self,
            constrained_path,
            data_points,
            kernel,
            num_particles,
            resample_threshold=0.5):

        self.constrained_path = constrained_path

        self.data_points = data_points

        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self.iteration = 0

        self.num_iterations = len(data_points)

    def sample(self):
        self._init_swarm()

        while self.iteration < self.num_iterations:
            #             print 'Iteration {0} of {1}.'.format(self.iteration, self.num_iterations)

            self._sample_new_particles()

            if self.iteration < self.num_iterations - 1:
                self._resample_if_necessary()

            self.iteration += 1

            assert self.constrained_path[self.iteration] is self.swarm.particles[0]

#             for i in range(1, self.swarm.num_particles):
#                 assert self.constrained_path[self.iteration] is not self.swarm.particles[i]

        return self.swarm

    def _init_swarm(self):
        self.swarm = ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        self.swarm.add_particle(uniform_weight, self.constrained_path[1])

        for _ in range(self.num_particles - 1):
            self.swarm.add_particle(uniform_weight, self._propose_particle(None))

        for particle in self.swarm.particles:
            assert particle.parent_particle is None

        self._resample_if_necessary()

        self.iteration += 1

    def _propose_particle(self, parent_particle):
        data_point = self.data_points[self.iteration]

        return self.kernel.propose_particle(data_point, parent_particle)

    def _resample_if_necessary(self):
        swarm = self.swarm

        if swarm.relative_ess <= self.resample_threshold:
            #             print 'Resampling {}'.format(swarm.relative_ess)

            new_swarm = ParticleSwarm()

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = np.random.multinomial(self.num_particles - 1, swarm.weights)

            assert not np.isneginf(self.constrained_path[self.iteration + 1].log_w)

            new_swarm.add_particle(log_uniform_weight, self.constrained_path[self.iteration + 1])

            for particle, multiplicity in zip(swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    assert not np.isneginf(particle.log_w)

                    new_swarm.add_particle(log_uniform_weight, particle)

        else:
            new_swarm = swarm

        self.swarm = new_swarm

    def _sample_new_particles(self):
        new_swarm = ParticleSwarm()

        particle = self.constrained_path[self.iteration + 1]

        parent_log_W = self.swarm.log_weights[0]

        new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        for parent_log_W, parent_particle in zip(self.swarm.log_weights[1:], self.swarm.particles[1:]):
            particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        self.swarm = new_swarm

    def _get_log_w(self, particle):
        if self.iteration < self.num_iterations - 1:
            return particle.log_w

        else:
            return particle.log_w - particle.state.log_p + particle.state.log_p_one
