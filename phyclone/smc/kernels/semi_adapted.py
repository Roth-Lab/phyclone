'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import random

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import Kernel


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
        """ Get the log probability of the state.
        """
        if self.parent_particle is None:
            log_q = 0

        elif state.node_idx in self.parent_particle.state.root_idxs:
            log_q = np.log(0.5) + self.log_q[state]

        else:
            old_num_roots = len(self.parent_particle.state.root_idxs)

            log_q = np.log(0.5) - old_num_roots * np.log(2)

        return log_q

    def sample_state(self):
        if self.parent_particle is None:
            state = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            u = random.random()

            if u < 0.5:
                q = np.exp(self.log_q.values())

                assert abs(1 - sum(q)) < 1e-6

                q = q / sum(q)

                idx = np.random.multinomial(1, q).argmax()

                state = self.log_q.keys()[idx]

            else:
                state = self._propose_new_node()

        return state

    def _init_dist(self):
        if self.parent_particle is None:
            return

        states = self._propose_existing_node()

        log_q = [x.log_p for x in states]

        log_q = log_normalize(np.array(log_q))

        self.log_q = dict(zip(states, log_q))

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
        num_roots = len(self.parent_particle.state.roots)

        num_children = random.randint(0, num_roots)

        children = random.sample(self.parent_particle.state.root_idxs, num_children)

        node_idx = max(self.parent_particle.state.roots.keys() + [-1, ]) + 1

        root_idxs = set(self.parent_particle.state.root_idxs - set(children))

        root_idxs.add(node_idx)

        return self.kernel.create_state(
            self.data_point,
            self.parent_particle,
            node_idx,
            root_idxs
        )


class SemiAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return SemiAdaptedProposal(data_point, self, parent_particle)
