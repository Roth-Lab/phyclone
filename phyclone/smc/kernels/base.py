from phyclone.tree import Tree
from collections import deque


class TreeHolder(object):
    # __slots__ = 'log_w', 'parent_particle', 'tree', 'data', '_tree'

    def __init__(self, tree, tree_dist):

        # self.data = data

        self._tree_dist = tree_dist

        self.log_p = 0

        self.tree = tree

        self._hash_val = 0

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    def copy(self):
        return TreeHolder(self.tree, self._tree_dist)
        # TODO: re-write this? building tree unnecessarily here

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self.log_p = self._tree_dist.log_p(tree)
        # self.tree_roots = tree.roots
        self._data = tree.data
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()

    @tree.getter
    def tree(self):
        return Tree.from_dict(self._data, self._tree)


class Particle(object):
    # __slots__ = 'log_w', 'parent_particle', 'tree', 'data', '_tree'

    def __init__(self, log_w, parent_particle, tree, data, tree_dist, perm_dist):
        self._built_tree = deque(maxlen=1)

        self.log_w = log_w

        self.parent_particle = parent_particle

        self.data = data

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        self.log_p = 0

        self.log_pdf = 0

        self.log_p_one = 0

        self.tree = tree

        self._hash_val = 0

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    def copy(self):
        cls = self.__class__

        new = cls.__new__(cls)

        new._built_tree = deque(maxlen=1)
        new.log_w = self.log_w
        new.parent_particle = self.parent_particle
        new.data = self.data.copy()
        new._tree_dist = self._tree_dist
        new._perm_dist = self._perm_dist
        new.log_p = self.log_p
        new.log_pdf = self.log_pdf
        new.log_p_one = self.log_p_one
        new._hash_val = self._hash_val
        new.tree_roots = self.tree_roots.copy()
        new._tree = self._tree.copy()
        return new
        # return Particle(self.log_w, self.parent_particle, self.tree, self.data, self._tree_dist, self._perm_dist)

    @property
    def tree(self):
        return self._tree

    @tree.getter
    def tree(self):
        return Tree.from_dict(self.data, self._tree)
        # return self._tree.copy()

    @tree.setter
    def tree(self, tree):
        self.log_p = self._tree_dist.log_p(tree)
        if self._perm_dist is None:
            self.log_pdf = 0.0
        else:
            self.log_pdf = self._perm_dist.log_pdf(tree)
        self.log_p_one = self._tree_dist.log_p_one(tree)
        self.tree_roots = tree.roots
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()
        # self._tree = tree

    @property
    def built_tree(self):
        return self._built_tree

    @built_tree.setter
    def built_tree(self, tree):
        self._built_tree.append(tree)

    @built_tree.getter
    def built_tree(self):
        return self._built_tree.pop()


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

    def create_particle(self, data_point, log_q, parent_particle, tree, data):
        """  Create a new particle from a parent particle.
        """
        # if self.perm_dist is None:
        #     if parent_particle is None:
        #         log_w = self._get_log_p(tree) - log_q
        #
        #     else:
        #         log_w = self._get_log_p(tree) - self._get_log_p(parent_particle.tree) - log_q
        #
        # else:
        #     if parent_particle is None:
        #         log_w = self._get_log_p(tree) + self.perm_dist.log_pdf(tree) - log_q
        #
        #     else:
        #         parent_tree = parent_particle.tree
        #         log_w = self._get_log_p(tree) - self._get_log_p(parent_tree) + \
        #             self.perm_dist.log_pdf(tree) - self.perm_dist.log_pdf(parent_tree) - \
        #             log_q
        particle = Particle(0, parent_particle, tree, data, self.tree_dist, self.perm_dist)

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
        # return Particle(log_w, parent_particle, tree, data, self.tree_dist, self.perm_dist)

    def propose_particle(self, data_point, parent_particle, data):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        tree = proposal_dist.sample()

        log_q = proposal_dist.log_p(tree)

        return self.create_particle(data_point, log_q, parent_particle, tree, data)

    def _get_log_p(self, tree):
        """ Compute joint distribution.
        """
        return self.tree_dist.log_p(tree)


class ProposalDistribution(object):
    """ Abstract class for proposal distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, parent_tree=None):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self._rng = kernel.rng

        self._set_parent_tree(parent_tree)

    def _empty_tree(self):
        """ Tree has no nodes
        """
        # return (self.parent_particle is None) or (len(self.parent_particle.tree.roots) == 0)
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
