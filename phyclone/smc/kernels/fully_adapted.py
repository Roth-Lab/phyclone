from __future__ import division

import itertools
import numpy as np
import random

from phyclone.math_utils import log_normalize, discrete_rvs
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
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
        return self._log_p[tree]

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        u = random.random()
        
        # First particle
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size)

            if u < (1 - self.outlier_proposal_prob):
                node = tree.create_root_node([])

                tree.add_data_point_to_node(self.data_point, node)

            else:
                tree.add_data_point_to_outliers(self.data_point)        
        
        # Particles t=2 ...
        # Only outliers in tree
        elif len(self.parent_particle.tree.nodes) == 0:
            if u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()
        
        p = np.exp(np.array(list(self._log_p.values())))

        idx = discrete_rvs(p)

        tree = list(self._log_p.keys())[idx]

        return tree

    def _init_dist(self):
        if self.parent_particle is None or len(self.parent_particle.tree.nodes) == 0:
            return
        
        trees = self._propose_existing_node() + self._propose_new_node()
        
        if self.outlier_proposal_prob > 0:
            trees.extend(self._propose_outlier_node())
        
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
        proposed_trees = []
        
        num_roots = len(self.parent_particle.tree.roots)

        for r in range(0, num_roots + 1):
            for children in itertools.combinations(self.parent_particle.tree.roots, r):
                tree = self.parent_particle.tree.copy()
                
                tree.create_root_node(children=children, data=[self.data_point])
                
                proposed_trees.append(tree)
        
        return proposed_trees.append(tree)

    def _propose_outlier_node(self):
        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return [tree]


class FullyAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposalDistribution(
            data_point, self, parent_particle, outlier_proposal_probs=self.outlier_proposal_prob
        )
