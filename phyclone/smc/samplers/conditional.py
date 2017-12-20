from __future__ import division, print_function

import networkx as nx
import numpy as np

from phyclone.smc.samplers.base import AbstractSMCSampler

import phyclone.smc.swarm


class ConditionalSMCSampler(AbstractSMCSampler):
    """ SMC sampler which conditions a fixed path.
    """

    def __init__(self, current_tree, data_points, kernel, num_particles, resample_threshold=0.5):
        super().__init__(data_points, kernel, num_particles, resample_threshold=resample_threshold)

        self.constrained_path = self._get_constrained_path(current_tree)

    def _get_constrained_path(self, tree):
        constrained_path = [None, ]

        data_to_node = tree.labels

        node_idx = 0

        old_to_new_node_idx = {-1: -1}

        root_idxs = set()

        for data_point in self.data_points:
            old_node_idx = data_to_node[data_point.idx]

            if old_node_idx not in old_to_new_node_idx:
                for child in tree.nodes[old_node_idx].children:
                    root_idxs.remove(old_to_new_node_idx[child.idx])

                old_to_new_node_idx[old_node_idx] = node_idx

                root_idxs.add(node_idx)

                node_idx += 1

            parent_particle = constrained_path[-1]

            proposal_dist = self.kernel.get_proposal_distribution(data_point, parent_particle)

            state = self.kernel.create_state(data_point, parent_particle, old_to_new_node_idx[old_node_idx], root_idxs)

            log_q = proposal_dist.log_p(state)

            particle = self.kernel.create_particle(data_point, log_q, parent_particle, state)

            constrained_path.append(particle)

        assert nx.is_isomorphic(tree.graph, particle.state.tree.graph)

        return constrained_path

    def _get_log_w(self, particle):
        if self.iteration < self.num_iterations - 1:
            return particle.log_w

        else:
            # Enforce that the sum of the tree is one and add auxillary term for permutation
            return particle.log_w - particle.state.log_p + particle.state.log_p_one + particle.state.tree.log_p_sigma

    def _init_swarm(self):
        self.swarm = phyclone.smc.swarm.ParticleSwarm()

        uniform_weight = -np.log(self.num_particles)

        self.swarm.add_particle(uniform_weight, self.constrained_path[1])

        for _ in range(self.num_particles - 1):
            self.swarm.add_particle(uniform_weight, self._propose_particle(None))

        for particle in self.swarm.particles:
            assert particle.parent_particle is None

        self.iteration += 1

    def _resample_swarm(self):
        if self.swarm.relative_ess <= self.resample_threshold:
            new_swarm = phyclone.smc.swarm.ParticleSwarm()

            log_uniform_weight = -np.log(self.num_particles)

            multiplicities = np.random.multinomial(self.num_particles - 1, self.swarm.weights)

            assert not np.isneginf(self.constrained_path[self.iteration + 1].log_w)

            new_swarm.add_particle(log_uniform_weight, self.constrained_path[self.iteration + 1])

            for particle, multiplicity in zip(self.swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    assert not np.isneginf(particle.log_w)

                    new_swarm.add_particle(log_uniform_weight, particle)

            self.swarm = new_swarm

    def _update_swarm(self):
        new_swarm = phyclone.smc.swarm.ParticleSwarm()

        particle = self.constrained_path[self.iteration + 1]

        parent_log_W = self.swarm.log_weights[0]

        new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        for parent_log_W, parent_particle in zip(self.swarm.log_weights[1:], self.swarm.particles[1:]):
            particle = self._propose_particle(parent_particle)

            new_swarm.add_particle(parent_log_W + self._get_log_w(particle), particle)

        self.swarm = new_swarm
