'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import random

from fscrp.kernels.marginal.base import MarginalKernel


class BootstrapProposal(object):
    """ Bootstrap proposal distribution.

    A simple proposal from the prior distribution.
    """

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

    def get_log_q(self, state):
        """ Get the log probability of the state.
        """
        if self.parent_particle is None:
            log_q = 0

        elif state.node_idx in self.parent_particle.state.root_idxs:
            num_roots = len(state.root_idxs)

            log_q = np.log(0.5) - np.log(num_roots)

        else:
            old_num_roots = len(self.parent_particle.state.root_idxs)

            log_q = np.log(0.5) - old_num_roots * np.log(2)

        return log_q

    def sample_state(self):
        """ Sample a new state from the proposal distribution.
        """
        if self.parent_particle is None:
            state = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            u = random.random()

            if u < 0.5:
                state = self._propose_existing_node()

            else:
                state = self._propose_new_node()

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
        num_roots = len(self.parent_particle.state.root_idxs)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.state.root_idxs, num_children)

        node_idx = max(self.parent_particle.state.nodes.keys() + [-1, ]) + 1

        root_idxs = self.parent_particle.state.root_idxs - set(children)

        root_idxs.add(node_idx)

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            root_idxs
        )


class MarginalBootstrapKernel(MarginalKernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return BootstrapProposal(data_point, self, parent_particle)
