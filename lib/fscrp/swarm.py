'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict
from pydp.utils import log_sum_exp

import numpy as np


class ParticleSwarm(object):

    def __init__(self):
        self.particles = []

        self._log_norm_const = None

        self._unnormalized_log_weights = []

    @property
    def ess(self):
        return 1 / np.sum(np.square(self.weights))

    @property
    def log_norm_const(self):
        if self._log_norm_const is None:
            self._log_norm_const = log_sum_exp(self.unnormalized_log_weights)

        return self._log_norm_const

    @property
    def log_weights(self):
        return self.unnormalized_log_weights - self.log_norm_const

    @property
    def num_particles(self):
        return len(self.particles)

    @property
    def relative_ess(self):
        return self.ess / self.num_particles

    @property
    def unnormalized_log_weights(self):
        return np.array(self._unnormalized_log_weights)

    @property
    def weights(self):
        weights = np.exp(self.log_weights)

        weights = weights / weights.sum()

        return weights

    def add_particle(self, log_weight, particle):
        '''
        Args:
            log_weight: Unnormalized log weight of particle
            particle: Particle
        '''
        self.particles.append(particle)

        self._unnormalized_log_weights.append(log_weight)

        self._log_norm_const = None

    def to_dict(self):
        result = defaultdict(float)

        for p, w in zip(self.particles, self.weights):
            result[p] += w

        return result
