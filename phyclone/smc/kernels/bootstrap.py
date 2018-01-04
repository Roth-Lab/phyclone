from __future__ import division

import numpy as np
import random

from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class BootstrapProposalDistribution(ProposalDistribution):
    """ Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.outlier_proposal_prob = outlier_proposal_prob

    def log_p(self, tree):
        """ Get the log probability of the tree.
        """
        if self.parent_particle is None:
            log_q = 0

        elif tree.labels[self.data_point.idx] == -1:
            log_q = np.log(self.outlier_proposal_prob)

        elif tree.labels[self.data_point.idx] in self.parent_particle.tree.nodes:
            num_nodes = len(self.parent_particle.tree.nodes)

            log_q = np.log((1 - self.outlier_proposal_prob) / 2) - np.log(num_nodes)

        else:
            old_num_roots = len(self.parent_particle.tree.roots)

            log_q = np.log((1 - self.outlier_proposal_prob) / 2) - old_num_roots * np.log(2)

        return log_q

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        if self.parent_particle is None:
            tree = Tree(self.kernel.alpha, self.kernel.grid_size)

            node = tree.create_root_node([])

            tree.add_data_point_to_node(self.data_point, node)

        else:
            u = random.random()

            if u < (1 - self.outlier_proposal_prob) / 2:
                tree = self._propose_existing_node()

            elif u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        return tree

    def _propose_existing_node(self):
        node = random.choice(list(self.parent_particle.tree.roots))

        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_node(self.data_point, node)

        return tree

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree.roots)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.tree.roots, num_children)

        tree = self.parent_particle.tree.copy()

        node = tree.create_root_node(children)

        tree.add_data_point_to_node(self.data_point, node)

        return tree

    def _propose_outlier(self):
        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return tree


class BootstrapKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposalDistribution(data_point, self, parent_particle)
