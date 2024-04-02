from phyclone.smc.swarm import Particle


class Kernel(object):
    """ Abstract class representing an SMC kernel targeting the marginal FS-CRP distribution.

    Subclasses should implement the get_proposal_distribution method.
    """

    @property
    def rng(self):
        return self._rng

    def get_proposal_distribution(self, data_point, parent_particle, parent_tree=None):
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

    def create_particle(self, log_q, parent_particle, tree):
        """  Create a new particle from a parent particle.
        """
        particle = Particle(0, parent_particle, tree, self.tree_dist, self.perm_dist)

        if self.perm_dist is None:
            if parent_particle is None:
                log_w = particle.log_p - log_q

            else:
                log_w = particle.log_p - parent_particle.log_p - log_q

        else:
            if parent_particle is None:
                log_w = particle.log_p + particle.log_pdf - log_q

            else:
                log_w = particle.log_p - parent_particle.log_p + \
                    particle.log_pdf - parent_particle.log_pdf - \
                    log_q

        particle.log_w = log_w
        return particle

    def propose_particle(self, data_point, parent_particle):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        tree = proposal_dist.sample()

        log_q = proposal_dist.log_p(tree)

        return self.create_particle(log_q, parent_particle, tree)

    def _get_log_p(self, tree):
        """ Compute joint distribution.
        """
        return self.tree_dist.log_p(tree)


class ProposalDistribution(object):
    """ Abstract class for proposal distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0.0, parent_tree=None):
        self.data_point = data_point

        self.tree_dist = kernel.tree_dist

        self.perm_dist = kernel.perm_dist

        self.outlier_proposal_prob = outlier_proposal_prob

        self.parent_particle = parent_particle

        self._rng = kernel.rng

        self._set_parent_tree(parent_tree)

    def _empty_tree(self):
        """ Tree has no nodes
        """
        return (self.parent_particle is None) or (len(self.parent_particle.tree_roots) == 0)

    def _set_parent_tree(self, parent_tree):
        if self.parent_particle is not None:
            if parent_tree is not None:
                self.parent_tree = parent_tree
            else:
                self.parent_tree = self.parent_particle.tree
        else:
            self.parent_tree = None

    def log_p(self, state):
        """ Get the log probability of proposing a tree.
        """
        raise NotImplementedError

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        raise NotImplementedError
