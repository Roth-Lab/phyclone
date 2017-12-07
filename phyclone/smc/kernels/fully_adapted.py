from __future__ import division

import itertools
import numpy as np

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import Kernel


class FullyAdaptedProposal(object):
    """ Fully adapted proposal density.

    Considers all possible proposals and weight according to log probability.
    """

    def __init__(self, data_point, kernel, parent_particle):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self._init_dist()

    def get_log_q(self, state):
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

        for r in range(0, num_roots + 1):
            for child_idxs in itertools.combinations(self.parent_particle.state.root_idxs, r):
                root_idxs = self.parent_particle.state.root_idxs - set(child_idxs)

                root_idxs.add(node_idx)

                proposed_states.append(
                    self.kernel.create_state(self.data_point, self.parent_particle, node_idx, root_idxs)
                )

        return proposed_states


class FullyAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return FullyAdaptedProposal(data_point, self, parent_particle)
