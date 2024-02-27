from phyclone.smc.kernels.base import Kernel
from phyclone.smc.kernels import FullyAdaptedKernel, SemiAdaptedKernel


class FlipKernel(Kernel):

    def __init__(self, tree_dist, rng, num_mutations, outlier_proposal_prob=0.0, perm_dist=None):
        super().__init__(tree_dist, rng, perm_dist=perm_dist)

        self.outlier_proposal_prob = outlier_proposal_prob

        self.fully_adapted_kernel = FullyAdaptedKernel(tree_dist, rng, outlier_proposal_prob=outlier_proposal_prob)

        self.semi_adapted_kernel = SemiAdaptedKernel(tree_dist, rng, outlier_proposal_prob=outlier_proposal_prob)

        self.num_mutations = num_mutations

    def get_proposal_distribution(self, data_point, parent_particle, parent_tree=None):
        num_nodes = 1
        if parent_tree is not None:
            num_nodes = parent_tree.get_number_of_nodes()
        elif parent_particle is not None:
            parent_tree = parent_particle.tree
            num_nodes = parent_tree.get_number_of_nodes()

        num_samples = data_point.grid_size[0]

        num_mutations = self.num_mutations

        if num_nodes >= 10 and num_samples >= 10 and num_mutations >= 1000:
            res = self.semi_adapted_kernel.get_proposal_distribution(data_point, parent_particle, parent_tree)
        else:
            res = self.fully_adapted_kernel.get_proposal_distribution(data_point, parent_particle, parent_tree)

        return res
