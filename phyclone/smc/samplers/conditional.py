import networkx as nx
import numpy as np

from phyclone.smc.samplers.base import AbstractSMCSampler
from phyclone.tree import Tree

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

        node_map = {}

        new_tree = Tree(tree.grid_size)

        parent_tree = None

        for data_point in self.data_points:
            new_tree = new_tree.copy()

            old_node = data_to_node[data_point.idx]

            if old_node == -1:
                new_tree.add_data_point_to_outliers(data_point)

            elif old_node in node_map:
                new_tree.add_data_point_to_node(data_point, node_map[old_node])

            else:
                children = []

                for child in tree.get_children(old_node):
                    children.append(node_map[child])

                new_node = new_tree.create_root_node(children)

                node_map[old_node] = new_node

                new_tree.add_data_point_to_node(data_point, new_node)

            parent_particle = constrained_path[-1]

            proposal_dist = self.kernel.get_proposal_distribution(data_point, parent_particle, parent_tree)

            log_q = proposal_dist.log_p(new_tree)

            particle = self.kernel.create_particle(data_point, log_q, parent_particle, new_tree, self.data_points)

            constrained_path.append(particle)

            parent_tree = new_tree

            # print('Constrained path build for dp {}, iter {}'.format(data_point.idx, len(constrained_path)-1))

        # assert nx.is_isomorphic(tree.graph, particle.tree.graph)
        assert nx.is_isomorphic(tree.graph, new_tree.graph)

        return constrained_path

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

            # multiplicities = np.random.multinomial(self.num_particles - 1, self.swarm.weights)
            multiplicities = self._rng.multinomial(self.num_particles - 1, self.swarm.weights)

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
