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

    def __init__(self, kernel='bootstrap', num_particles=10, outlier_proposal_prob=0, resample_threshold=0.5):

        kernels = {
            'bootstrap': phyclone.smc.kernels.BootstrapKernel,
            'fully-adapted': phyclone.smc.kernels.FullyAdaptedKernel,
            'semi-adapted': phyclone.smc.kernels.SemiAdaptedKernel
        }

        self.kernel_cls = kernels[kernel]

        self.num_particles = num_particles

        self.outlier_proposal_prob = outlier_proposal_prob

        self.resample_threshold = resample_threshold

    def sample_tree(self, tree):
        """ Sample a new tree
        """
        swarm = self.sample_swarm(tree)

        return self._sample_tree_from_swarm(swarm)

    def sample_swarm(self, tree):
        """ Sample a new SMC swarm
        """
        perm_dist = phyclone.smc.utils.PermutationDistribution()

        data_sigma = perm_dist.sample(tree)

        kernel = self.kernel_cls(
            tree.alpha, tree.grid_size, outlier_proposal_prob=self.outlier_proposal_prob, perm_dist=perm_dist,
        )

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

    def sample_tree(self, tree):
        labels = []

        for label in tree.labels.values():
            if label != -1:
                labels.append(label)

        subtree_root_child = random.choice(labels)

        subtree_root = tree.get_parent(subtree_root_child)

        parent = tree.get_parent(subtree_root)

        subtree = tree.get_subtree(subtree_root)

        tree.remove_subtree(subtree)

        for data_point in tree.outliers:
            tree.remove_data_point_from_outliers(data_point)

            subtree.add_data_point_to_outliers(data_point)

        swarm = self.sample_swarm(subtree)

        swarm = self._correct_weights(parent, swarm, tree)

        return self._sample_tree_from_swarm(swarm)

    # TODO: Check that this targets the correct distribution.
    # Specifically do we need a term for the random choice of node.
    def _correct_weights(self, parent, swarm, tree):
        """ Correct weights so target is the distribtuion on the full tree
        """
        new_swarm = phyclone.smc.swarm.ParticleSwarm()

        for p, w in zip(swarm.particles, swarm.unnormalized_log_weights):
            subtree = p.tree

            w -= subtree.log_p_one

            new_tree = tree.copy()

            new_tree.add_subtree(subtree, parent=parent)

            for data_point in subtree.outliers:
                new_tree.add_data_point_to_outliers(data_point)

            new_tree.update()

            w += new_tree.log_p_one

            p.tree = new_tree

            new_swarm.add_particle(w, p)

        return new_swarm
