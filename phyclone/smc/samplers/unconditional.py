from phyclone.smc.samplers import SMCSampler
from phyclone.smc.utils import RootPermutationDistribution
from phyclone.utils.math import discrete_rvs


class UnconditionalSMCSampler(object):
    __slots__ = ("kernel", "num_particles", "resample_threshold", "_rng")

    def __init__(self, kernel, num_particles=20, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self._rng = kernel.rng

    def sample_tree(self, tree):
        data_sigma = RootPermutationDistribution.sample(tree, self._rng)

        smc_sampler = SMCSampler(
            data_sigma,
            self.kernel,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold,
        )

        swarm = smc_sampler.sample()

        idx = discrete_rvs(swarm.weights, self._rng)

        return swarm.particles[idx].tree
