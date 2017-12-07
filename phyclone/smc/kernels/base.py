from __future__ import division, print_function

from collections import namedtuple

import numpy as np

from phyclone.math_utils import log_factorial
from phyclone.smc.utils import get_num_data_points_per_node
from phyclone.tree import MarginalNode


Particle = namedtuple('Particle', ['log_w', 'parent_particle', 'state'])


class State(object):
    """ A partial state of the SMC algorithm.

    This class stores the partially constructed tree during the SMC.
    """

    def __init__(self, nodes, node_idx, log_p_prior, root_idxs):
        self.log_p_prior = log_p_prior

        assert node_idx < len(nodes)

        self.nodes = nodes

        self.node_idx = node_idx

        self._root_idxs = tuple(sorted(root_idxs))

        self._dummy_root = None

        self._log_p = None

        self._log_p_one = None

    def __key(self):
        return (self.node_idx, self._root_idxs)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def dummy_root(self):
        """ A node connecting all concrete rootnodes in the tree.
        """
        if self._dummy_root is None:
            self._dummy_root = MarginalNode(-1, self.nodes.values()[0].grid_size, children=self.root_nodes)

        return self._dummy_root

    @property
    def log_p(self):
        """ Log joint probability of the state marginalizing over value of dummy root.
        """
        if self._log_p is None:
            self._log_p = self.dummy_root.log_p

        return self.log_p_prior + self._log_p

    @property
    def log_p_one(self):
        """ Log joint probability of the state with dummy root having value of one.
        """
        if self._log_p_one is None:
            self._log_p_one = self.dummy_root.log_p_one

        return self.log_p_prior + self._log_p_one

    @property
    def root_idxs(self):
        """ List indexes for concrete root nodes in the tree.
        """
        return set(self._root_idxs)

    @property
    def root_nodes(self):
        """ List of concrete root nodes.
        """
        return [self.nodes[idx] for idx in self._root_idxs]


class Kernel(object):
    """ Abstract class representing an SMC kernel targeting the marginal FS-CRP distribution.

    Sub-classes should implement the get_proposal_distribution method.
    """

    def get_proposal_distribution(self, data_point, parent_particle):
        """ Get proposal distribution given the current data point and parent particle.
        """
        raise NotImplementedError

    def __init__(self, alpha, grid_size):
        """
        Parameters
        ----------
        alpha: float
            Concentration parameter of the CRP.
        grid_size: int
            The size of the grid to approximate the recursion integrals.
        """
        self.alpha = alpha

        self.grid_size = grid_size

    def create_particle(self, data_point, log_q, parent_particle, state):
        """  Create a new particle from a parent particle.
        """
        if parent_particle is None:
            log_w = state.log_p - log_q

        else:
            log_w = state.log_p - parent_particle.state.log_p - log_q

        return Particle(log_w, parent_particle, state)

    def create_state(self, data_point, parent_particle, node_idx, root_idxs):
        """ Create a new state.

        Parameters
        ----------
        data_point: array_like (float)
            Current data point.
        parent_particle: Particle
            Parent particle in genealogy.
        node_idx: int
            Index of the node the data point is assigned to.
        root_idxs: array_like (int)
            List of indexes for concrete nodes.
        """
        log_p_prior = self._compute_log_p_prior(data_point, parent_particle, node_idx, root_idxs)

        if parent_particle is None:
            assert node_idx == 0

            assert root_idxs == set([0, ])

            nodes = {}

            nodes[node_idx] = MarginalNode(node_idx, self.grid_size)

        elif node_idx in parent_particle.state.nodes:
            assert root_idxs == parent_particle.state.root_idxs

            nodes = parent_particle.state.nodes.copy()

            nodes[node_idx] = parent_particle.state.nodes[node_idx].copy()

        else:
            child_idxs = parent_particle.state.root_idxs - root_idxs

            nodes = parent_particle.state.nodes.copy()

            children = [parent_particle.state.nodes[idx] for idx in child_idxs]

            nodes[node_idx] = MarginalNode(node_idx, self.grid_size, children=children)

        nodes[node_idx].add_data_point(data_point)

        return State(nodes, node_idx, log_p_prior, root_idxs)

    def propose_particle(self, data_point, parent_particle):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        state = proposal_dist.sample_state()

        log_q = proposal_dist.get_log_q(state)

        return self.create_particle(data_point, log_q, parent_particle, state)

    def _compute_log_p_prior(self, data_point, parent_particle, node_idx, root_idxs):
        """ Compute the incremental FS-CRP prior contribution.
        """
        if parent_particle is None:
            log_p = np.log(self.alpha)

        elif node_idx in parent_particle.state.root_idxs:
            node_counts = get_num_data_points_per_node(parent_particle)

            log_p = np.log(node_counts[node_idx])

        else:
            child_idxs = parent_particle.state.root_idxs - root_idxs

            node_counts = get_num_data_points_per_node(parent_particle)

            num_nodes = len(parent_particle.state.nodes)

            # CRP prior
            log_p = np.log(self.alpha)

            # Tree prior
            log_p += (num_nodes - 1) * np.log(num_nodes + 1) - num_nodes * np.log(num_nodes + 2)

            log_perm_norm = log_factorial(sum([node_counts[idx] for idx in child_idxs]))

            for idx in child_idxs:
                log_perm_norm -= log_factorial(node_counts[idx])

            log_p -= log_perm_norm

        return log_p
