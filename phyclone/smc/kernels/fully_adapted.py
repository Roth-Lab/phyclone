import itertools
import numpy as np

from phyclone.math_utils import log_normalize, discrete_rvs
from phyclone.smc.kernels.base import Kernel, ProposalDistribution
from phyclone.tree import Tree


class FullyAdaptedProposalDistribution(ProposalDistribution):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    def __init__(self, data_point, kernel, parent_particle, factorial_arr, outlier_proposal_prob=0.0):
        super().__init__(data_point, kernel, parent_particle, factorial_arr)
        
        self.outlier_proposal_prob = outlier_proposal_prob
        
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
        self._log_p = {}

        trees = self._get_existing_node_trees() + self._get_new_node_trees()
        
        if self.outlier_proposal_prob > 0:
            trees.extend(self._get_outlier_tree())
        
        log_q = np.array([self.kernel.tree_dist.log_p(x) for x in trees])

        log_q = log_normalize(log_q)

        self._log_p = dict(zip(trees, log_q))

    def _get_existing_node_trees(self):
        """ Enumerate all trees obtained by adding the data point to an existing node.
        """
        trees = []
        
        if self.parent_particle is None:
            return trees
        
        nodes = self.parent_particle.tree.roots

        for node in nodes:
            tree = self.parent_particle.tree.copy()

            tree.add_data_point_to_node(self.data_point, node)

            trees.append(tree)

        return trees

    def _get_new_node_trees(self):
        """ Enumerate all trees obtained by adding the data point to a new node.
        """
        trees = []
        
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size, self.factorial_arr, self.memo_logs)
            
            tree.create_root_node(children=[], data=[self.data_point])
            
            trees.append(tree)
        
        else:
            num_roots = len(self.parent_particle.tree.roots)
    
            for r in range(0, num_roots + 1):
                for children in itertools.combinations(self.parent_particle.tree.roots, r):
                    tree = self.parent_particle.tree.copy()
                    
                    tree.create_root_node(children=children, data=[self.data_point])
                    
                    trees.append(tree)
        
        return trees

    def _get_outlier_tree(self):
        """ Get the tree obtained by adding data point as outlier
        """
        if self.parent_particle is None:
            tree = Tree(self.data_point.grid_size, self.factorial_arr, self.memo_logs)
        
        else:
            tree = self.parent_particle.tree.copy()

        tree.add_data_point_to_outliers(self.data_point)

        return [tree]


class FullyAdaptedKernel(Kernel):

    def __init__(self, tree_prior_dist, factorial_arr, memo_logs, outlier_proposal_prob=0, perm_dist=None):
        super().__init__(tree_prior_dist, factorial_arr, memo_logs, perm_dist=perm_dist)

        self.outlier_proposal_prob = outlier_proposal_prob

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposalDistribution(
            data_point,
            self,
            parent_particle,
            self.factorial_arr,
            outlier_proposal_prob=self.outlier_proposal_prob
        )
