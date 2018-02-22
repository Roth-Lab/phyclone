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

    Sub-classes should implement the get_proposal_distribution method.
    """

    def get_proposal_distribution(self, data_point, parent_particle):
        """ Get proposal distribution given the current data point and parent particle.
        """
        raise NotImplementedError

    def __init__(self, alpha, grid_size, outlier_proposal_prob=0, perm_dist=None, propose_roots=True):
        """
        Parameters
        ----------
        alpha: float
            Concentration parameter of the CRP.
        grid_size: int
            The size of the grid to approximate the recursion integrals.
        outlier_proposal_prob: float
            Probability of proposing an outlier.
        perm_dist: PermutationDistribution
            The permutation distribution used in a particle Gibbs sampler to reorder data points. Set to None if single
            pass SMC is being performed.
        propose_roots: bool
            Determines whether to propose adding data points to existing nodes that are roots or alternatively adding to
            any existing node.
        """
        self.alpha = alpha

        self.grid_size = grid_size

        self.outlier_proposal_prob = outlier_proposal_prob

        self.perm_dist = perm_dist

        self.propose_roots = propose_roots

    def create_particle(self, data_point, log_q, parent_particle, tree):
        """  Create a new particle from a parent particle.
        """
        if self.perm_dist is None:
            if parent_particle is None:
                log_w = tree.log_p - log_q

            else:
                log_w = tree.log_p - parent_particle.tree.log_p - log_q

        else:
            if parent_particle is None:
                log_w = tree.log_p + self.perm_dist.log_pdf(tree) - log_q

            else:
                log_w = tree.log_p + self.perm_dist.log_pdf(tree) - \
                    parent_particle.tree.log_p - self.perm_dist.log_pdf(parent_particle.tree) - log_q

        return Particle(log_w, parent_particle, tree)

    def propose_particle(self, data_point, parent_particle):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        tree = proposal_dist.sample()

        log_q = proposal_dist.log_p(tree)

        return self.create_particle(data_point, log_q, parent_particle, tree)


class ProposalDistribution(object):
    """ Abstract class for proposal distribution.
    """

    def log_p(self, state):
        """ Get the log probability of a tree.
        """
        raise NotImplementedError

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        raise NotImplementedError
