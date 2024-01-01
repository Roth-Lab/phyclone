from sklearn.metrics import homogeneity_completeness_v_measure

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.mcmc.metropolis_hastings import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.smc.kernels import SemiAdaptedKernel
from phyclone.tree import FSCRPDistribution, Tree

from toy_data import load_test_data
from phyclone.math_utils import simple_log_factorial
from math import inf
import numpy as np

data, true_tree = load_test_data(cluster_size=5)

factorial_arr = np.full(len(data)+1, -inf)
simple_log_factorial(len(data), factorial_arr)

tree = Tree.get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01)

mh_sampler = PruneRegraphSampler()

kernel = SemiAdaptedKernel(FSCRPDistribution(1.0))

pg_sampler = ParticleGibbsTreeSampler(kernel)

for i in range(1000):
    tree = pg_sampler.sample_tree(tree)

    tree = mh_sampler.sample_tree(tree)

    if i % 10 == 0:
        pred_labels = [tree.labels[x] for x in sorted(tree.labels)]
        true_labels = [true_tree.labels[x] for x in sorted(true_tree.labels)]
        print(i)
        print(homogeneity_completeness_v_measure(true_labels, pred_labels), len(tree.nodes))
        print(tree.log_p)
