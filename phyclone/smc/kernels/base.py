class Particle(object):
    __slots__ = 'log_w', 'parent_particle', 'tree'

    def __init__(self, log_w, parent_particle, tree):
        self.log_w = log_w

        self.parent_particle = parent_particle

        self.tree = tree

    def copy(self):
        return Particle(self.log_w, self.parent_particle, self.tree.copy())


class Kernel(object):
    """ Abstract class representing an SMC kernel targeting the marginal FS-CRP distribution.

    Subclasses should implement the get_proposal_distribution method.
    """

    @property
    def rng(self):
        return self._rng

    def get_proposal_distribution(self, data_point, parent_particle):
        """ Get proposal distribution given the current data point and parent particle.
        """
        raise NotImplementedError

    def __init__(self, tree_dist, rng, perm_dist=None):
        """
        Parameters
        ----------
        tree_dist: TreeJointDistribution
            Joint distribution of tree
        # outlier_proposal_prob: float
        #     Probability of proposing an outlier.
        perm_dist: PermutationDistribution
            The permutation distribution used in a particle Gibbs sampler to reorder data points. Set to None if single
            pass SMC is being performed.
        """
        self.tree_dist = tree_dist

        self.perm_dist = perm_dist

        self._rng = rng

    def create_particle(self, data_point, log_q, parent_particle, tree):
        """  Create a new particle from a parent particle.
        """
        if self.perm_dist is None:
            if parent_particle is None:
                log_w = self._get_log_p(tree) - log_q

            else:
                log_w = self._get_log_p(tree) - self._get_log_p(parent_particle.tree) - log_q

        else:
            if parent_particle is None:
                log_w = self._get_log_p(tree) + self.perm_dist.log_pdf(tree) - log_q

            else:
                log_w = self._get_log_p(tree) - self._get_log_p(parent_particle.tree) + \
                    self.perm_dist.log_pdf(tree) - self.perm_dist.log_pdf(parent_particle.tree) - \
                    log_q

        return Particle(log_w, parent_particle, tree)

    def propose_particle(self, data_point, parent_particle):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        tree = proposal_dist.sample()

        log_q = proposal_dist.log_p(tree)

        return self.create_particle(data_point, log_q, parent_particle, tree)

    def _get_log_p(self, tree):
        """ Compute joint distribution.
        """
        return self.tree_dist.log_p(tree)


class ProposalDistribution(object):
    """ Abstract class for proposal distribution.
    """

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self._rng = kernel.rng

    def _empty_tree(self):
        """ Tree has no nodes
        """
        return (self.parent_particle is None) or (len(self.parent_particle.tree.roots) == 0)

    def log_p(self, state):
        """ Get the log probability of proposing a tree.
        """
        raise NotImplementedError

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        raise NotImplementedError
