import numpy as np
import random

from phyclone.math_utils import log_binomial_coefficient
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class BootstrapProposalDistribution(ProposalDistribution):
    """ Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, factorial_arr, outlier_proposal_prob=0.0):
        super().__init__(data_point, kernel, parent_particle, factorial_arr)

        self.outlier_proposal_prob = outlier_proposal_prob
        
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
                num_nodes = len(self.parent_particle.tree.roots)
                
                log_p = np.log((1 - self.outlier_proposal_prob) / 2) - np.log(num_nodes)
            
            # New node
            else:
                old_num_roots = len(self.parent_particle.tree.roots)
                
                log_p = np.log((1 - self.outlier_proposal_prob) / 2)
                
                if old_num_roots > 0:
                    num_children = len(tree.get_children(node))
                
                    log_p -= np.log(old_num_roots + 1) + log_binomial_coefficient(old_num_roots, num_children)
        
        return log_p

    def sample(self):
        """ Sample a new tree from the proposal distribution.
        """
        # u = random.random()
        u = self._rng.random()
        
        # First particle
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size, self.factorial_arr, self.memo_logs)

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
        nodes = self.parent_particle.tree.roots
   
        # node = random.choice(list(nodes))
        node = self._rng.choice(list(nodes))

        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_node(self.data_point, node)

        return tree

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.tree.roots)

        # num_children = random.randint(0, num_roots)
        num_children = self._rng.integers(0, num_roots+1)

        # children = random.sample(self.parent_particle.tree.roots, num_children)
        children = self._rng.choice(self.parent_particle.tree.roots, num_children, replace=False)

        tree = self.parent_particle.tree.copy()

        tree.create_root_node(children=children, data=[self.data_point])

        return tree

    def _propose_outlier(self):
        tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return tree


class BootstrapKernel(Kernel):

    def __init__(self, tree_prior_dist, factorial_arr, memo_logs, rng, outlier_proposal_prob=0, perm_dist=None):
        super().__init__(tree_prior_dist, factorial_arr, memo_logs, rng, perm_dist=perm_dist)

        self.outlier_proposal_prob = outlier_proposal_prob

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposalDistribution(
            data_point,
            self,
            parent_particle,
            self.factorial_arr,
            outlier_proposal_prob=self.outlier_proposal_prob
        )
