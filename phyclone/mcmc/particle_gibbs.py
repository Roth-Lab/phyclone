from __future__ import division

import random

import phyclone.math_utils
import phyclone.smc.kernels
import phyclone.smc.samplers.particle_gibbs
import phyclone.smc.samplers.swarm
import phyclone.smc.utils


class ParticleGibbsTreeSampler(object):
    """ Particle Gibbs sampler targeting sampling a full tree.
    """

    def __init__(
            self,
            grid_size,
            alpha=1.0,
            kernel='bootstrap',
            num_particles=10,
            resample_threshold=0.5):

        if kernel == 'bootstrap':
            self.kernel = phyclone.smc.kernels.BootstrapKernel(alpha, grid_size)

        elif kernel == 'fully-adapted':
            self.kernel = phyclone.smc.kernels.FullyAdaptedKernel(alpha, grid_size)

        elif kernel == 'semi-adapted':
            self.kernel = phyclone.smc.kernels.SemiAdaptedKernel(alpha, grid_size)

        else:
            raise Exception('Unrecognized kernel: {}'.format(kernel))

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

    @property
    def alpha(self):
        return self.kernel.alpha

    @alpha.setter
    def alpha(self, x):
        self.kernel.alpha = x

    def sample_tree(self, data, tree):
        """ Sample a new tree
        """
        swarm = self.sample_swarm(data, tree)

        return self._sample_tree_from_swarm(swarm)

    def sample_swarm(self, data, tree):
        """ Sample a new SMC swarm
        """
        data_sigma = phyclone.smc.utils.sample_sigma(tree)

#         data_sigma = [data[data_idx] for data_idx in sigma]

        for i, d in enumerate(data):
            assert i == d.idx

        sampler = phyclone.smc.samplers.particle_gibbs.ParticleGibbsSampler(
            tree,
            data_sigma,
            self.kernel,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold
        )

        return sampler.sample()

    def _sample_tree_from_swarm(self, swarm):
        """ Given an SMC swarm sample a tree
        """
        particle_idx = phyclone.math_utils.discrete_rvs(swarm.weights)

        particle = swarm.particles[particle_idx]

        return particle.state.tree


class ParticleGibbsSubtreeSampler(ParticleGibbsTreeSampler):
    """ Particle Gibbs sampler which resamples a sub-tree.
    """

    def sample_tree(self, data, tree):
        node_idxs = list(tree.nodes.keys())

        subtree_root_idx = random.choice(node_idxs)

        parent_node = tree.get_parent_node(tree.nodes[subtree_root_idx])

        if parent_node.idx == 'root':
            parent_node = None

        subtree = tree.get_subtree(tree.nodes[subtree_root_idx])

        tree.remove_subtree(subtree)

        subtree.relabel_nodes(0)

        for x in tree.outliers:
            if random.random() < 0.5:
                subtree.outliers.append(x)

                tree.outliers.remove(x)

        swarm = self.sample_swarm(data, subtree)

        if len(tree.nodes) == 0:
            min_node_idx = 0

        else:
            min_node_idx = max(tree.nodes.keys()) + 1

        swarm = self._correct_weights(min_node_idx, parent_node, swarm, tree)

        sub_tree = self._sample_tree_from_swarm(swarm)

        sub_tree.relabel_nodes(min_node_idx)

        tree.add_subtree(sub_tree, parent=parent_node)

        tree.outliers.extend(sub_tree.outliers)

        tree.relabel_nodes(0)

        return tree

    # TODO: Check that this targets the correct distribution.
    # Specifically do we need a term for the random choice of node.
    def _correct_weights(self, min_node_idx, parent_node, swarm, tree):
        """ Correct weights so target is the distribtuion on the full tree
        """
        new_swarm = phyclone.smc.samplers.swarm.ParticleSwarm()

        for p, w in zip(swarm.particles, swarm.unnormalized_log_weights):
            w -= p.state.log_p_one

            sub_tree = p.state.tree

            sub_tree.relabel_nodes(min_value=min_node_idx)

            tree.add_subtree(sub_tree, parent=parent_node)

            tree.outliers.extend(sub_tree.outliers)

            w += tree.log_p_one

            new_swarm.add_particle(w, p)

            tree.remove_subtree(sub_tree)

            for x in sub_tree.outliers:
                tree.outliers.remove(x)

        return new_swarm
