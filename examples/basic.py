from sklearn.metrics import homogeneity_completeness_v_measure

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.gibbs_mh import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.smc.kernels import SemiAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution

from toy_data import load_test_data
from phyclone.run import instantiate_and_seed_RNG

rng = instantiate_and_seed_RNG(0, None)

tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

data, true_tree = load_test_data(rng, cluster_size=5)

tree = Tree.get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng)

mh_sampler = PruneRegraphSampler(tree_dist, rng)

kernel = SemiAdaptedKernel(tree_dist, rng)

pg_sampler = ParticleGibbsTreeSampler(kernel, rng)

for i in range(1000):
    tree = pg_sampler.sample_tree(tree)

    tree = mh_sampler.sample_tree(tree)

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        true_labels = [true_tree.labels[x] for x in sorted(true_tree.labels)]
        print(i)
        print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
        print(tree_dist.log_p(tree))
