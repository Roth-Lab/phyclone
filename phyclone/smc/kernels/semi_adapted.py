import numpy as np
from phyclone.utils.math import log_binomial_coefficient, log_normalize
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree
from functools import lru_cache
from phyclone.smc.swarm import TreeHolder


class SemiAdaptedProposalDistribution(ProposalDistribution):
    """ Semi adapted proposal density.

    Considers all possible choice of existing nodes and one option for a new node proposed at random. This
    should provide a computational advantage over the fully adapted proposal.
    """
    __slots__ = "_log_p"

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0.0, parent_tree=None):
        super().__init__(data_point, kernel, parent_particle, outlier_proposal_prob, parent_tree)

        self._init_dist()

    def log_p(self, tree):
        """ Get the log probability of proposing the tree.
        """
        # node = tree.labels[self.data_point.idx]

        if self._empty_tree():
            log_p = self._get_log_p(tree)

        else:

            node = tree.labels[self.data_point.idx]

            assert node == tree.node_last_added_to

            # Existing node
            if node in self.parent_particle.tree_nodes or node == -1:
                log_p = np.log(0.5) + self._get_log_p(tree)

            # New node
            else:
                # old_num_roots = len(self.parent_particle.tree_roots)
                #
                # log_p = np.log(0.5)
                #
                # num_children = tree.get_number_of_children(node)
                #
                # log_p -= np.log(old_num_roots + 1) + log_binomial_coefficient(old_num_roots, num_children)
                old_num_roots = len(self.parent_particle.tree_roots)

                log_p = np.log(0.5)

                # num_children = tree.get_number_of_children(node)
                num_children = tree.num_children_on_node_that_matters

                log_p -= np.log(old_num_roots + 1) + log_binomial_coefficient(old_num_roots, num_children)

        return log_p

    def _get_log_p(self, tree):
        """ Get the log probability of the given tree. From stored dict, using TreeHolder intermediate.
        """
        if isinstance(tree, Tree):
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)
        else:
            tree_particle = tree
        return self._log_p[tree_particle]

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        if self._empty_tree():
            tree = self._propose_existing_node()
        else:
            u = self._rng.random()

            if u < 0.5:
                tree = self._propose_existing_node()
            else:
                tree = self._propose_new_node()

        return tree

    def _init_dist(self):
        self._log_p = {}
        trees = self._get_existing_node_trees()

        if self.outlier_proposal_prob > 0:
            trees.append(self._get_outlier_tree())

        if self._empty_tree():
            if self.parent_particle is None:
                tree = Tree(self.data_point.grid_size)
            else:
                tree = self.parent_tree.copy()

            tree.create_root_node(children=[], data=[self.data_point])
            tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

            trees.append(tree_particle)

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

    def _get_outlier_tree(self):
        """ Get the tree obtained by adding data point as outlier
        """
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

        else:
            tree = self.parent_tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        tree_particle = TreeHolder(tree, self.tree_dist, self.perm_dist)

        return tree_particle

    def _propose_existing_node(self):
        q = np.exp(list(self._log_p.values()))

        assert abs(1 - sum(q)) < 1e-6

        q = q / sum(q)

        idx = self._rng.multinomial(1, q).argmax()

        tree = list(self._log_p.keys())[idx]

        return tree

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree_roots)

        num_children = self._rng.integers(0, num_roots + 1)

        children = self._rng.choice(self.parent_particle.tree_roots, num_children, replace=False)

        tree = self.parent_particle.tree

        tree.create_root_node(children=children, data=[self.data_point])

        tree_container = TreeHolder(tree, self.tree_dist, self.perm_dist)

        return tree_container

        # return tree


class SemiAdaptedKernel(Kernel):
    __slots__ = "outlier_proposal_prob"

    def __init__(self, tree_dist, rng, outlier_proposal_prob=0.0, perm_dist=None):
        super().__init__(tree_dist, rng, perm_dist=perm_dist)

        self.outlier_proposal_prob = outlier_proposal_prob

    def get_proposal_distribution(self, data_point, parent_particle, parent_tree=None):
        if parent_particle is not None:
            parent_particle.built_tree = parent_tree
        return _get_cached_semi_proposal_dist(data_point, self, parent_particle, self.outlier_proposal_prob,
                                              self.tree_dist.prior.alpha)


@lru_cache(maxsize=1024)
def _get_cached_semi_proposal_dist(data_point, kernel, parent_particle, outlier_proposal_prob, alpha):
    if parent_particle is not None:
        ret = SemiAdaptedProposalDistribution(
            data_point,
            kernel,
            parent_particle,
            outlier_proposal_prob=outlier_proposal_prob,
            parent_tree=parent_particle.built_tree
        )
    else:
        ret = SemiAdaptedProposalDistribution(
            data_point,
            kernel,
            parent_particle,
            outlier_proposal_prob=outlier_proposal_prob,
            parent_tree=None
        )
    return ret
