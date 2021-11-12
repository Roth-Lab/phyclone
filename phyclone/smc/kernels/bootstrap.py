from __future__ import division

import numpy as np
import random

from phyclone.math_utils import log_binomial_coefficient
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class BootstrapProposalDistribution(ProposalDistribution):
    """ Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0, propose_roots=True):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.outlier_proposal_prob = outlier_proposal_prob
        
        self.propose_roots = propose_roots

    def log_p(self, tree):
        """ Get the log probability of the tree.
        """
        # First particle
        if self.parent_particle is None:
            if tree.labels[self.data_point.idx] == -1:
                log_p = np.log(self.outlier_proposal_prob)

            else:
                log_p = np.log(1 - self.outlier_proposal_prob)
        # Particles t=2 ...        
        else:
            node = tree.labels[self.data_point.idx]
            
            # Outlier
            if node == -1:
                log_p = np.log(self.outlier_proposal_prob)
            
            # Node in tree
            elif node in self.parent_particle.tree.nodes:
                if self.propose_roots:
                    num_nodes = len(self.parent_particle.tree.roots)
                
                else:
                    num_nodes = len(self.parent_particle.tree.nodes)
    
                log_p = np.log((1 - self.outlier_proposal_prob) / 2) - np.log(num_nodes)
            
            # New node
            else:
                old_num_roots = len(self.parent_particle.tree.roots)
                
                log_p = np.log((1 - self.outlier_proposal_prob) / 2)
                
                if old_num_roots > 0:
                    num_children = len(tree.get_children(node))
                
                    log_p -= np.log(old_num_roots) + log_binomial_coefficient(old_num_roots, num_children)
        
        return log_p

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
        
        # Nodes in the tree        
        else:
            if u < (1 - self.outlier_proposal_prob) / 2:
                tree = self._propose_existing_node()

            elif u < (1 - self.outlier_proposal_prob):
                tree = self._propose_new_node()

            else:
                tree = self._propose_outlier()

        return tree

    def _propose_existing_node(self):
        if self.propose_roots:
            nodes = self.parent_particle.tree.roots

        else:
            nodes = self.parent_particle.tree.nodes
   
        node = random.choice(list(nodes))

        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_node(self.data_point, node)

        return tree

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


class BootstrapKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposalDistribution(
            data_point,
            self,
            parent_particle,
            outlier_proposal_prob=self.outlier_proposal_prob,
            propose_roots=self.propose_roots
        )
