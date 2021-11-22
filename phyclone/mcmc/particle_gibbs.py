import random

import phyclone.math_utils
import phyclone.smc.samplers
import phyclone.smc.swarm
import phyclone.smc.utils


class ParticleGibbsTreeSampler(object):
    """ Particle Gibbs sampler targeting sampling a full tree.
    """

    def __init__(self, kernel, num_particles=10, resample_threshold=0.5):
        self.kernel = kernel

        self.num_particles = num_particles

        self.resample_threshold = resample_threshold

    def sample_tree(self, tree):
        """ Sample a new tree
        """
        swarm = self.sample_swarm(tree)

        return self._sample_tree_from_swarm(swarm)

    def sample_swarm(self, tree):
        """ Sample a new SMC swarm
        """
        perm_dist = phyclone.smc.utils.RootPermutationDistribution()

        data_sigma = perm_dist.sample(tree)

        sampler = phyclone.smc.samplers.ConditionalSMCSampler(
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

        return particle.tree


class ParticleGibbsSubtreeSampler(ParticleGibbsTreeSampler):
    """ Particle Gibbs sampler which resamples a sub-tree.
    """

    def sample_tree(self, tree):
        nodes = []

        for label in tree.labels.values():
            if label != -1:
                nodes.append(label)

        subtree_root_child = random.choice(nodes)

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

            w -= self.kernel.tree_dist.log_p_one(subtree)

            new_tree = tree.copy()

            new_tree.add_subtree(subtree, parent=parent)

            for data_point in subtree.outliers:
                new_tree.add_data_point_to_outliers(data_point)

            new_tree.update()

            w += self.kernel.tree_dist.log_p_one(new_tree)

            p.tree = new_tree

            new_swarm.add_particle(w, p)

        return new_swarm
