'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import random

from phyclone.math_utils import log_normalize
from phyclone.smc.kernels.base import Kernel, ProposalDistribution


class SemiAdaptedProposalDistribution(ProposalDistribution):
    """ Semi adapted proposal density.

    Considers all possible choice of existing nodes and one option for a new node proposed at random. This
    should provide a computational advantage over the fully adapted proposal.
    """

    def __init__(self, data_point, kernel, parent_particle, outlier_proposal_prop=0):
        self.data_point = data_point

        self.kernel = kernel

        self.parent_particle = parent_particle

        self.outlier_proposal_prop = 0

        self._init_dist()

    def log_p(self, state):
        """ Get the log probability of the state.
        """
        if self.parent_particle is None:
            log_q = 0

        elif state.node_idx == -1:
            log_q = np.log(self.outlier_proposal_prop)

        elif state.node_idx in self.log_q:
            log_q = np.log((1 - self.outlier_proposal_prop) / 2) + self.log_q[state.node_idx][0]

        else:
            old_num_roots = len(self.parent_particle.state.root_idxs)

            log_q = np.log((1 - self.outlier_proposal_prop) / 2) - old_num_roots * np.log(2)

        return log_q

    def sample(self):
        if self.parent_particle is None:
            state = self.kernel.create_state(self.data_point, self.parent_particle, 0, set([0, ]))

        else:
            u = random.random()

            if u < (1 - self.outlier_proposal_prop) / 2:
                q = np.exp([x[0] for x in self.log_q.values()])

                assert abs(1 - sum(q)) < 1e-6

                q = q / sum(q)

                idx = np.random.multinomial(1, q).argmax()

                state = list(self.log_q.values())[idx][1]

            elif u < (1 - self.outlier_proposal_prop):
                state = self._propose_new_node()

            else:
                state = self._propose_outlier()

        return state

    def _init_dist(self):
        if self.parent_particle is None:
            return

        states = self._propose_existing_node()

        log_q = [x.log_p for x in states]

        log_q = log_normalize(np.array(log_q))

        self.log_q = {}

        for log_q_state, state in zip(log_q, states):
            self.log_q[state.node_idx] = (log_q_state, state)

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


class SemiAdaptedKernel(Kernel):

    def get_proposal_distribution(self, data_point, parent_particle):
        return SemiAdaptedProposalDistribution(data_point, self, parent_particle)
