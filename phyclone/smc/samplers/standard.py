from __future__ import division

import numpy as np

from phyclone.smc.samplers.base import AbstractSMCSampler

import phyclone.smc.swarm


class SMCSampler(AbstractSMCSampler):
    """ Standard SMC sampler with adaptive resampling.
    """

    def _get_log_w(self, particle):
        if self.iteration < self.num_iterations - 1:
            return particle.log_w

        else:
            # Enforce that the sum of the tree is one
            return particle.log_w - particle.state.log_p + particle.state.log_p_one

    def _init_swarm(self):
        self.swarm = phyclone.smc.swarm.ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        for _ in range(self.num_particles):
            self.swarm.add_particle(uniform_weight, None)

    def _resample_swarm(self):
        if self.swarm.relative_ess <= self.resample_threshold:
            new_swarm = phyclone.smc.swarm.ParticleSwarm()

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = np.random.multinomial(self.num_particles, self.swarm.weights)

            for particle, multiplicity in zip(self.swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    new_swarm.add_particle(log_uniform_weight, particle)

            self.swarm = new_swarm

    def _update_swarm(self):
        new_swarm = phyclone.smc.swarm.ParticleSwarm()

        for parent_log_W, parent_particle in zip(self.swarm.log_weights, self.swarm.particles):
            particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        self.swarm = new_swarm