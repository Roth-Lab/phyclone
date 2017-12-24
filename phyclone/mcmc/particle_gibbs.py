from __future__ import division

import random

import phyclone.math_utils
import phyclone.smc.kernels
import phyclone.smc.samplers
import phyclone.smc.swarm
import phyclone.smc.utils


class ParticleGibbsTreeSampler(object):
    """ Particle Gibbs sampler targeting sampling a full tree.
    """

    def __init__(self, kernel='bootstrap', num_particles=10, resample_threshold=0.5):

        kernels = {
            'bootstrap': phyclone.smc.kernels.BootstrapKernel,
            'fully-adapted': phyclone.smc.kernels.FullyAdaptedKernel,
            'semi-adapted': phyclone.smc.kernels.SemiAdaptedKernel
        }

        self.kernel_cls = kernels[kernel]

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

    def sample_tree(self, data, tree):
        """ Sample a new tree
        """
        swarm = self.sample_swarm(data, tree)

        return self._sample_tree_from_swarm(swarm)

    def sample_swarm(self, data, tree):
        """ Sample a new SMC swarm
        """
        data_sigma = phyclone.smc.utils.sample_sigma(tree)

        for i, d in enumerate(data):
            assert i == d.idx

        kernel = self.kernel_cls(tree.alpha, tree.grid_size)

        sampler = phyclone.smc.samplers.ConditionalSMCSampler(
            tree,
            data_sigma,
            kernel,
            num_particles=self.num_particles,
            resample_threshold=self.resample_threshold
        )

        return sampler.sample()

    def _sample_tree_from_swarm(self, swarm):
        """ Given an SMC swarm sample a tree
        """
        particle_idx = phyclone.math_utils.discrete_rvs(swarm.weights)

        particle = swarm.particles[particle_idx]

        return particle.tree


class ParticleGibbsSubtreeSampler(ParticleGibbsTreeSampler):
    """ Particle Gibbs sampler which resamples a sub-tree.
    """

    def sample_tree(self, data, tree):
        if phyclone.math_utils.bernoulli_rvs(0.25):
            return self._sample_tree_from_swarm(self.sample_swarm(data, tree))

        subtree_root = random.choice(list(tree.nodes.values()))

        parent_node = tree.get_parent_node(subtree_root)

        if parent_node is None:
            parent_idx = None

        else:
            parent_idx = parent_node.idx

        subtree = tree.get_subtree(subtree_root)

        tree.remove_subtree(subtree)

        for x in tree.outliers:
            if random.random() < 0.5:
                subtree.outliers.append(x)

                tree.outliers.remove(x)

        swarm = self.sample_swarm(data, subtree)

        swarm = self._correct_weights(parent_idx, swarm, tree)

        return self._sample_tree_from_swarm(swarm)

    # TODO: Check that this targets the correct distribution.
    # Specifically do we need a term for the random choice of node.
    def _correct_weights(self, parent_idx, swarm, tree):
        """ Correct weights so target is the distribtuion on the full tree
        """
        new_swarm = phyclone.smc.swarm.ParticleSwarm()

        for p, w in zip(swarm.particles, swarm.unnormalized_log_weights):
            subtree = p.tree

            w -= subtree.log_p_one

            new_tree = tree.copy()

            new_tree.add_subtree(subtree, parent_idx=parent_idx)

            new_tree.outliers.extend(subtree.outliers)

            new_tree.update_likelihood()

            new_tree.relabel_nodes()

            w += new_tree.log_p_one

            p.tree = new_tree

            new_swarm.add_particle(w, p)

        return new_swarm
