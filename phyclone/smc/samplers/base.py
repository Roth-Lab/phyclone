class AbstractSMCSampler(object):
    """Abstract class for an SMC sampler."""

    __slots__ = (
        "data_points",
        "kernel",
        "num_particles",
        "_rng",
        "resample_threshold",
        "iteration",
        "num_iterations",
        "swarm",
    )

    def __init__(self, data_points, kernel, num_particles, resample_threshold=0.5):
        self.data_points = data_points

        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

        self.iteration = 0

        self.num_iterations = len(data_points)

        self.swarm = None

        self._rng = kernel.rng

    def sample(self):
        self._init_swarm()

        self._resample_swarm()

        while self.iteration < self.num_iterations:
            self._update_swarm()

            if self.iteration < self.num_iterations - 1:
                self._resample_swarm()

            self.iteration += 1

        return self.swarm

    def _propose_particle(self, parent_particle):
        data_point = self.data_points[self.iteration]

        return self.kernel.propose_particle(data_point, parent_particle)

    def _get_log_w(self, particle):
        if self.iteration < self.num_iterations - 1:
            return particle.log_w

        else:
            # Enforce that the sum of the tree is one and add auxillary term for permutation
            return particle.log_w - particle.log_p + particle.log_p_one

    def _init_swarm(self):
        """Initialize `self.swarm` by adding first data point."""
        raise NotImplementedError

    def _resample_swarm(self):
        """Resample a new `self.swarm` with uniform weights if relative ESS drops below `self.resample_threshold`."""
        raise NotImplementedError

    def _update_swarm(self):
        """Update `self.swarm` by adding next data point."""
        raise NotImplementedError
