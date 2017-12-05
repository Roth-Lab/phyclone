'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import random

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import MarginalKernel


class SemiAdaptedProposal(object):
    """ Semi adapted proposal density.

    Considers all possible choice of existing nodes and one option for a new node proposed at random. This
    should provide a computational advantage over the fully adapted proposal.
    """

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self._init_dist()

    def get_log_q(self, state):
        # Hack for the conditional path
        if state not in self.states:
            return 0

        return self.log_q[self.states.index(state)]

    def sample_state(self):
        q = np.exp(self.log_q)

        assert abs(1 - sum(q)) < 1e-6

        q = q / sum(q)

        idx = np.random.multinomial(1, q).argmax()

        return self.states[idx]

    def _init_dist(self):
        self.states = self._propose_new_node()

        if self.parent_particle is not None:
            self.states.extend(self._propose_existing_node())

        log_q = [x.log_p for x in self.states]

        self.log_q = log_normalize(np.array(log_q))

    def _propose_existing_node(self):
        proposed_states = []

        for node_idx in self.parent_particle.state.root_idxs:
            proposed_states.append(
                self.kernel.create_state(
                    self.data_point,
                    self.parent_particle,
                    node_idx,
                    self.parent_particle.state.root_idxs
                )
            )

        return proposed_states

    def _propose_new_node(self):
        if self.parent_particle is None:
            return [
                self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))
            ]

        proposed_states = []

        node_idx = max(self.parent_particle.state.nodes.keys() + [-1, ]) + 1

        num_roots = len(self.parent_particle.state.root_idxs)

        r = random.randint(0, num_roots)

        child_idxs = random.sample(self.parent_particle.state.root_idxs, r)

        root_idxs = self.parent_particle.state.root_idxs - set(child_idxs)

        root_idxs.add(node_idx)

        proposed_states.append(
            self.kernel.create_state(self.data_point, self.parent_particle, node_idx, root_idxs)
        )

        return proposed_states


class MarginalSemiAdaptedKernel(MarginalKernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return SemiAdaptedProposal(data_point, self, parent_particle)
