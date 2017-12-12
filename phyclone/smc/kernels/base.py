from __future__ import division, print_function

from collections import namedtuple
from scipy.misc import logsumexp as log_sum_exp

import numpy as np

from phyclone.math_utils import log_factorial
from phyclone.smc.utils import get_num_data_points_per_node
from phyclone.tree import MarginalNode


Particle = namedtuple('Particle', ['log_w', 'parent_particle', 'state'])


class State(object):
    """ A partial state of the SMC algorithm.

    This class stores the partially constructed tree during the SMC.
    """

    def __init__(self, node_idx, log_p_prior, outliers, roots):
        self.log_p_prior = log_p_prior

        self.node_idx = node_idx

        self.outliers = outliers

        self.roots = roots

        self._dummy_root = None

        self._log_p = None

        self._log_p_one = None

        assert (node_idx in self.root_idxs) or (node_idx == -1)

    def __key(self):
        return (self.node_idx, self.root_idxs)

    def __eq__(self, y):
        return self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def dummy_root(self):
        """ A node connecting all concrete rootnodes in the tree.
        """
        if len(self.roots) == 0:
            return None

        if self._dummy_root is None:
            grid_size = self.root_nodes[0].grid_size

            self._dummy_root = MarginalNode(-1, grid_size, children=self.roots.values())

        return self._dummy_root

    @property
    def log_p(self):
        """ Log joint probability of the state marginalizing over value of dummy root.
        """
        if self._log_p is None:
            if self.dummy_root is None:
                self._log_p = 0

            else:
                self._log_p = self.dummy_root.log_p

            for data_point in self.outliers:
                log_norm = np.log(data_point.value.shape[1])

                self._log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return self.log_p_prior + self._log_p

    @property
    def log_p_one(self):
        """ Log joint probability of the state with dummy root having value of one.
        """
        if self._log_p_one is None:
            if self.dummy_root is None:
                self._log_p_one = 0

            else:
                self._log_p_one = self.dummy_root.log_p_one

            for data_point in self.outliers:
                log_norm = np.log(data_point.value.shape[1])

                self._log_p += np.sum(log_sum_exp(data_point.value - log_norm, axis=1))

        return self.log_p_prior + self._log_p_one

    @property
    def root_idxs(self):
        return frozenset([x.idx for x in self.roots.values()])

    @property
    def root_nodes(self):
        return list(self.roots.values())


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
            outliers = []

            roots = {}

            if node_idx == -1:
                outliers.append(data_point)

            else:
                assert node_idx == 0

                assert root_idxs == set([0, ])

                roots[0] = MarginalNode(node_idx, self.grid_size)

                roots[node_idx].add_data_point(data_point)

        else:
            outliers = list(parent_particle.state.outliers)

            if node_idx == -1:
                outliers.append(data_point)

                roots = parent_particle.state.roots.copy()

            elif node_idx in parent_particle.state.roots:
                assert root_idxs == parent_particle.state.root_idxs

                roots = parent_particle.state.roots.copy()

                roots[node_idx] = roots[node_idx].shallow_copy()

                roots[node_idx].add_data_point(data_point)

            else:
                child_idxs = parent_particle.state.root_idxs - root_idxs

                children = [parent_particle.state.roots[idx] for idx in child_idxs]

                roots = {}

                for idx in root_idxs:
                    if idx in parent_particle.state.root_idxs:
                        roots[idx] = parent_particle.state.roots[idx]

                roots[node_idx] = MarginalNode(node_idx, self.grid_size, children=children)

                roots[node_idx].add_data_point(data_point)

        return State(node_idx, log_p_prior, outliers, roots)

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
        if node_idx == -1:
            log_p = np.log(1e-10)

        else:
            if parent_particle is None:
                log_p = np.log(self.alpha)

            elif node_idx in parent_particle.state.root_idxs:
                node_counts = get_num_data_points_per_node(parent_particle)

                log_p = np.log(node_counts[node_idx])

            else:
                child_idxs = parent_particle.state.root_idxs - root_idxs

                node_counts = get_num_data_points_per_node(parent_particle)

                num_nodes = len(set(node_counts.keys()))

                # CRP prior
                log_p = np.log(self.alpha)

                # Tree prior
                log_p += (num_nodes - 1) * np.log(num_nodes + 1) - num_nodes * np.log(num_nodes + 2)

                log_perm_norm = log_factorial(sum([node_counts[idx] for idx in child_idxs]))

                for idx in child_idxs:
                    log_perm_norm -= log_factorial(node_counts[idx])

                log_p -= log_perm_norm

                log_p += np.log(1 - 1e-10)

        return log_p
