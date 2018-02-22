import numpy as np
import random

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class SemiAdaptedProposalDistribution(ProposalDistribution):
    """ Semi adapted proposal density.

    Considers all possible choice of existing nodes and one option for a new node proposed at random. This
    should provide a computational advantage over the fully adapted proposal.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0, propose_roots=True):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.outlier_proposal_prob = outlier_proposal_prob

        self.propose_roots = propose_roots

        self._init_dist()

    def log_p(self, tree):
        """ Get the log probability of the tree.
        """
        if self.parent_particle is None:
            log_p = 0

        elif tree.labels[self.data_point.idx] == -1:
            log_p = np.log(self.outlier_proposal_prob)

        elif tree.labels[self.data_point.idx] in self.parent_particle.tree.nodes:
            log_p = np.log((1 - self.outlier_proposal_prob) / 2) + self._log_p[tree]

        else:
            old_num_roots = len(self.parent_particle.tree.roots)

            log_p = np.log((1 - self.outlier_proposal_prob) / 2) - old_num_roots * np.log(2)

        return log_p

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        u = random.random()

        if self.parent_particle is None:
            tree = Tree(self.kernel.alpha, self.kernel.grid_size)

            if u < (1 - self.outlier_proposal_prob):
                node = tree.create_root_node([])

                tree.add_data_point_to_node(self.data_point, node)

            else:
                tree.add_data_point_to_outliers(self.data_point)

        # Only outliers in tree
        elif len(self.parent_particle.tree.nodes) == 0:
            if u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        else:
            if u < (1 - self.outlier_proposal_prob) / 2:
                q = np.exp(list(self._log_p.values()))

                assert abs(1 - sum(q)) < 1e-6

                q = q / sum(q)

                idx = np.random.multinomial(1, q).argmax()

                tree = list(self._log_p.keys())[idx]

            elif u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        return tree

    def _init_dist(self):
        if self.parent_particle is None:
            return

        trees = self._propose_existing_node()

        log_q = np.array([x.log_p for x in trees])

        log_q = log_normalize(log_q)

        self._log_p = dict(zip(trees, log_q))

    def _propose_existing_node(self):
        proposed_trees = []

        if self.propose_roots:
            nodes = self.parent_particle.tree.roots

        else:
            nodes = self.parent_particle.tree.nodes

        for node in nodes:
            tree = self.parent_particle.tree.copy()

            tree.add_data_point_to_node(self.data_point, node)

            proposed_trees.append(tree)

        return proposed_trees

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree.roots)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.tree.roots, num_children)

        tree = self.parent_particle.tree.copy()

        tree.create_root_node(children=children, data=[self.data_point])

        return tree

    def _propose_outlier(self):
        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return tree


class SemiAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return SemiAdaptedProposalDistribution(
            data_point,
            self, parent_particle,
            outlier_proposal_prob=self.outlier_proposal_prob,
            propose_roots=self.propose_roots
        )
