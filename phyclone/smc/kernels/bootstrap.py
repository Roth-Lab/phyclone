from __future__ import division

import numpy as np
import random

from phyclone.smc.kernels.base import Kernel, ProposalDistribution


class BootstrapProposalDistribution(ProposalDistribution):
    """ Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prob=0):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.outlier_proposal_prob = outlier_proposal_prob

    def log_p(self, state):
        """ Get the log probability of the state.
        """
        if self.parent_particle is None:
            log_q = 0

        elif state.node_idx == -1:
            log_q = np.log(self.outlier_proposal_prob)

        elif state.node_idx in self.parent_particle.state.root_idxs:
            num_roots = len(state.root_idxs)

            log_q = np.log((1 - self.outlier_proposal_prob) / 2) - np.log(num_roots)

        else:
            old_num_roots = len(self.parent_particle.state.root_idxs)

            log_q = np.log((1 - self.outlier_proposal_prob) / 2) - old_num_roots * np.log(2)

        return log_q

    def sample(self):
        """ Sample a new state from the proposal distribution.
        """
        if self.parent_particle is None:
            state = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            u = random.random()

            if len(self.parent_particle.state.roots) == 0:
                if u < (1 - self.outlier_proposal_prob):
                    state = self._propose_new_node()

                else:
                    state = self._propose_outlier()

            else:
                if u < (1 - self.outlier_proposal_prob) / 2:
                    state = self._propose_existing_node()

                elif u < (1 - self.outlier_proposal_prob):
                    state = self._propose_new_node()

                else:
                    state = self._propose_outlier()

        return state

    def _propose_existing_node(self):
        node_idx = random.choice(list(self.parent_particle.state.root_idxs))

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            self.parent_particle.state.root_idxs
        )

    def _propose_new_node(self):
        num_roots = len(self.parent_particle.state.roots)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.state.root_idxs, num_children)

        node_idx = max(list(self.parent_particle.state.root_idxs) + [-1, ]) + 1

        root_idxs = set(self.parent_particle.state.root_idxs - set(children))

        root_idxs.add(node_idx)

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            root_idxs
        )

    def _propose_outlier(self):
        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            -1,
            self.parent_particle.state.root_idxs
        )


class BootstrapKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposalDistribution(data_point, self, parent_particle)
