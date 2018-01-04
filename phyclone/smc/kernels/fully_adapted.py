from __future__ import division

import itertools
import numpy as np

from phyclone.math_utils import log_normalize, discrete_rvs
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    def __init__(self, data_point, kernel, parent_particle, use_outliers=False):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.use_outliers = use_outliers

        self._init_dist()

    def log_p(self, tree):
        """ Get the log probability of the tree.
        """
        return self._log_p[tree]

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        p = np.exp(np.array(list(self._log_p.values())))

        idx = discrete_rvs(p)

        tree = list(self._log_p.keys())[idx]

        return tree

    def _init_dist(self):
        self.trees = []

        if self.parent_particle is None:
            tree = Tree(self.kernel.alpha, self.kernel.grid_size)

            node = tree.create_root_node([])

            tree.add_data_point_to_node(self.data_point, node)

            self.trees.append(tree)

        else:
            self._propose_new_node()

            if self.use_outliers:
                self._propose_outlier_node()

            if self.parent_particle is not None:
                self._propose_existing_node()

        log_p = [t.log_p for t in self.trees]

        log_p = log_normalize(np.array(log_p))

        self._log_p = dict(zip(self.trees, log_p))

    def _propose_existing_node(self):
        for node in self.parent_particle.tree.roots:
            tree = self.parent_particle.tree.copy()

            tree.add_data_point_to_node(self.data_point, node)

            self.trees.append(tree)

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree.roots)

        for r in range(0, num_roots + 1):
            for children in itertools.combinations(self.parent_particle.tree.roots, r):
                tree = self.parent_particle.tree.copy()

                node = tree.create_root_node(children)

                tree.add_data_point_to_node(self.data_point, node)

                self.trees.append(tree)

    def _propose_outlier_node(self):
        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        self.trees.append(tree)


class FullyAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposalDistribution(data_point, self, parent_particle)
