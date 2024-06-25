import numpy as np

from phyclone.utils.math import log_sum_exp


class ParticleSwarm(object):
    """A collection of particles from one iteration of an SMC algorithm.

    This class handles tracking particles, their weights and normalization.
    """

    __slots__ = ("particles", "_log_norm_const", "_unnormalized_log_weights")

    def __init__(self):
        self.particles = []

        self._log_norm_const = None

        self._unnormalized_log_weights = []

    @property
    def ess(self):
        """Effective sample size of swarm."""
        return 1 / np.sum(np.square(self.weights))

    @property
    def log_norm_const(self):
        """Log of the normalization constant i.e. log(Z)."""
        if self._log_norm_const is None:
            self._log_norm_const = log_sum_exp(self.unnormalized_log_weights)

        return self._log_norm_const

    @property
    def log_weights(self):
        """Normalized log weights of particles."""
        return self.unnormalized_log_weights - self.log_norm_const

    @property
    def num_particles(self):
        """Number of particles in the swarm."""
        return len(self.particles)

    @property
    def relative_ess(self):
        """ESS normalized to number of particles."""
        return self.ess / self.num_particles

    @property
    def unnormalized_log_weights(self):
        """Raw log weights of particles."""
        return np.array(self._unnormalized_log_weights)

    @property
    def weights(self):
        """Particle weights."""
        weights = np.exp(self.log_weights)

        weights = weights / weights.sum()

        return weights

    def add_particle(self, log_weight, particle):
        """Add a particle to the swarm.

        Parameters
        ----------
        log_weight: float
            Unnormalized log weight of particle

        particle: Particle
        """
        self.particles.append(particle)

        self._unnormalized_log_weights.append(log_weight)

        self._log_norm_const = None

    def to_list(self):
        """Return a list of two tuples where the first entry is a particle and the second is the weight."""
        return zip(self.particles, self.weights)
