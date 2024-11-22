"""
Created on 8 Dec 2016

@author: Andrew Roth
"""

import numpy as np
from scipy.stats import gamma, beta, bernoulli


class GammaPriorConcentrationSampler(object):
    """Gibbs update assuming a gamma prior on the concentration parameter."""

    __slots__ = ("a", "b", "_rng")

    def __init__(self, a, b, rng):
        """
        Parameters
        ----------
        a: float
            Shape parameter of the gamma prior.
        b: float
            Rate parameter of the gamma prior.
        """
        self.a = a

        self.b = b

        self._rng = rng

    def sample(self, old_value, num_clusters, num_data_points):
        if num_clusters == 0:
            new_value = gamma.rvs(self.a, scale=(1 / self.b), random_state=self._rng)

        else:
            a = self.a

            b = self.b

            k = num_clusters

            n = num_data_points

            eta = beta.rvs(a=old_value + 1, b=n, random_state=self._rng)

            shape = a + k - 1

            rate = b - np.log(eta)

            x = shape / (n * rate)

            pi = x / (1 + x)

            shape += bernoulli.rvs(pi, random_state=self._rng)

            new_value = gamma.rvs(shape, scale=(1 / rate), random_state=self._rng)

            new_value = max(new_value, 1e-10)  # Catch numerical error

        return new_value
