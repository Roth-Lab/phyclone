import itertools
import numpy as np
from functools import lru_cache
from phyclone.utils.math import log_normalize, discrete_rvs
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.smc.swarm import TreeHolder
from phyclone.tree import Tree


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """
    __slots__ = "_log_p"

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0.0, parent_tree=None):
        super().__init__(data_point, kernel, parent_particle, outlier_proposal_prob, parent_tree)

        self._init_dist()

    def log_p(self, tree):
        """ Get the log probability of the tree.
        """
        if isinstance(tree, Tree):
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
        else:
            tree_particle = tree
        return self._log_p[tree_particle]

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        p = np.exp(np.array(list(self._log_p.values())))

        idx = discrete_rvs(p, self._rng)

        tree = list(self._log_p.keys())[idx]

        return tree

    def _init_dist(self):
        self._log_p = {}
        trees = self._get_existing_node_trees() + self._get_new_node_trees()

        if self.outlier_proposal_prob > 0:
            trees.extend(self._get_outlier_tree())
        log_q = np.array([x.log_p for x in trees])

        log_q = log_normalize(log_q)

        self._log_p = dict(zip(trees, log_q))

        self.parent_tree = None

    def _get_existing_node_trees(self):
        """ Enumerate all trees obtained by adding the data point to an existing node.
        """
        trees = []

        if self.parent_particle is None:
            return trees

        nodes = self.parent_particle.tree_roots

        for node in nodes:
            tree = self.parent_tree.copy()
            tree.add_data_point_to_node(self.data_point, node)
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
            trees.append(tree_particle)

        return trees

    def _get_new_node_trees(self):
        """ Enumerate all trees obtained by adding the data point to a new node.
        """
        trees = []

        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

            tree.create_root_node(children=[], data=[self.data_point])
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

            trees.append(tree_particle)

        else:
            num_roots = len(self.parent_particle.tree_roots)

            for r in range(0, num_roots + 1):
                for children in itertools.combinations(self.parent_particle.tree_roots, r):
                    tree = self.parent_tree.copy()

                    tree.create_root_node(children=children, data=[self.data_point])
                    tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

                    trees.append(tree_particle)

        return trees

    def _get_outlier_tree(self):
        """ Get the tree obtained by adding data point as outlier
        """
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

        else:
            tree = self.parent_tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

        return [tree_particle]


class FullyAdaptedKernel(Kernel):
    __slots__ = "outlier_proposal_prob"

    def __init__(self, tree_prior_dist, rng, outlier_proposal_prob=0, perm_dist=None):
        super().__init__(tree_prior_dist, rng, perm_dist=perm_dist)

        self.outlier_proposal_prob = outlier_proposal_prob

    def get_proposal_distribution(self, data_point, parent_particle, parent_tree=None):
        if parent_particle is not None:
            parent_particle.built_tree = parent_tree
        return _get_cached_full_proposal_dist(data_point, self, parent_particle, self.outlier_proposal_prob,
                                              self.tree_dist.prior.alpha)


@lru_cache(maxsize=1024)
def _get_cached_full_proposal_dist(data_point, kernel, parent_particle, outlier_proposal_prob, alpha):
    if parent_particle is not None:
        ret = FullyAdaptedProposalDistribution(
            data_point,
            kernel,
            parent_particle,
            outlier_proposal_prob=outlier_proposal_prob,
            parent_tree=parent_particle.built_tree
        )
    else:
        ret = FullyAdaptedProposalDistribution(
            data_point,
            kernel,
            parent_particle,
            outlier_proposal_prob=outlier_proposal_prob,
            parent_tree=None
        )
    return ret
