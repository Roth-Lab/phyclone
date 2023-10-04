import numpy as np
import random

from phyclone.math_utils import exp_normalize, discrete_rvs

import phyclone.math_utils


class DataPointSampler(object):
    """ Gibbs sample a new assignment for a data point.
    
    TODO: Confirm this is valid since we have a special condition to avoid creating empty nodes.
    """

    def __init__(self, tree_dist, rng: np.random.Generator, outliers=False):
        self.tree_dist = tree_dist
        
        self.outliers = outliers

        self._rng = rng

    def sample_tree(self, tree):
            data_idxs = list(tree.labels.keys())
            
            # random.shuffle(data_idxs)

            self._rng.shuffle(data_idxs)
            
            for data_idx in data_idxs:
                if len(tree.node_data[tree.labels[data_idx]]) > 1:
                    tree = self._sample_tree(data_idx, tree)
    
            return tree 
        
    def _sample_tree(self, data_idx, tree):
        data_point = tree.data[data_idx]
        
        old_node = tree.labels[data_idx]
        
        new_trees = []
        
        for new_node in tree.nodes:
            new_tree = tree.copy()
            
            new_tree.remove_data_point_from_node(data_point, old_node)
            
            new_tree.add_data_point_to_node(data_point, new_node)
            
            new_trees.append(new_tree)
        
        if self.outliers:
            new_tree = tree.copy()
            
            new_tree.remove_data_point_from_node(data_point, old_node)
            
            new_tree.add_data_point_to_outliers(data_point)
            
            new_trees.append(new_tree)
            
        log_q = np.array([self.tree_dist.log_p(x) for x in new_trees])

        log_q = phyclone.math_utils.log_normalize(log_q)
        
        q = np.exp(log_q)
        
        q = q / sum(q)

        # tree_idx = np.random.multinomial(1, q).argmax()
        tree_idx = self._rng.multinomial(1, q).argmax()
        
        return new_trees[tree_idx]  


class PruneRegraphSampler(object):
    """ Prune a subtree and regraph by Gibbs sampling possible attachement points
    """

    def __init__(self, tree_dist, rng: np.random.Generator):
        self.tree_dist = tree_dist

        self._rng = rng

    def sample_tree(self, tree):
        if len(tree.nodes) <= 1:
            return tree

        new_tree = tree.copy()

        # subtree_root = random.choice(new_tree.nodes)
        subtree_root = self._rng.choice(new_tree.nodes)

        subtree = new_tree.get_subtree(subtree_root)

        new_tree.remove_subtree(subtree)

        remaining_nodes = new_tree.nodes

        if len(remaining_nodes) == 0:
            return tree
        
        # TODO: Is this double counting the original tree
        trees = [tree]
        
        # Descendant from dummy normal node
        remaining_nodes.append(None)

        for parent in remaining_nodes:
            new_tree = tree.copy()

            subtree = new_tree.get_subtree(subtree_root)

            new_tree.remove_subtree(subtree)

            new_tree.add_subtree(subtree, parent=parent)

            new_tree.update()

            trees.append(new_tree)

        log_p = np.array([self.tree_dist.log_p(x) for x in trees])

        p, _ = exp_normalize(log_p)

        idx = discrete_rvs(p, self._rng)

        return trees[idx]

