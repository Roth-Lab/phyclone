'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from fscrp.data_structures import Particle
from fscrp.swarm import ParticleSwarm
from fscrp.particle_utils import iter_particles


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

            self._resample_if_necessary()

            self.iteration += 1

            assert self.constrained_path[self.iteration] is self.swarm.particles[0]

#             for i in range(1, self.swarm.num_particles):
#                 assert self.constrained_path[self.iteration] is not self.swarm.particles[i]

        return self.swarm.to_dict()

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
            new_swarm = ParticleSwarm()

#             print self.iteration, 'Resampling', swarm.relative_ess

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = np.random.multinomial(self.num_particles - 1, swarm.weights)

            assert not np.isneginf(self.constrained_path[self.iteration + 1].log_w)

            new_swarm.add_particle(log_uniform_weight, self.constrained_path[self.iteration + 1])

            for particle, multiplicity in zip(swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    assert not np.isneginf(particle.log_w)

                    new_particle = particle

                    new_swarm.add_particle(log_uniform_weight, new_particle)

        else:
            new_swarm = swarm

        self.swarm = new_swarm

    def _sample_new_particles(self):
        new_swarm = ParticleSwarm()

        for parent_log_W, parent_particle in zip(self.swarm.log_weights, self.swarm.particles):
            if parent_particle is self.constrained_path[self.iteration]:
                particle = self.constrained_path[self.iteration + 1]

            else:
                particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + particle.log_w, particle)

        self.swarm = new_swarm
