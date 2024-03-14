import numpy as np

from phyclone.concentration import GammaPriorConcentrationSampler
from phyclone.data.base import DataPoint
from phyclone.mcmc.gibbs_mh import PruneRegraphSampler
from phyclone.mcmc.particle_gibbs import ParticleGibbsTreeSampler
from phyclone.smc.samplers import UnconditionalSMCSampler
from phyclone.run import instantiate_and_seed_RNG
from phyclone.smc.kernels import FullyAdaptedKernel
from phyclone.tree import (
    FSCRPDistribution,
    UniformFSCRPDistribution,
    Tree,
    TreeJointDistribution,
)
from phyclone.process_trace import count_topology, _create_topology_dataframe
from phyclone.math_utils import log_binomial_likelihood

# data = "binomial"
data = "point_mass"

if data == "binomial":
    d = 5000
    g = 101
    grid = np.array([x / g for x in range(g)])
    grid[0] = 1e-100

    d_0 = np.array(
        [log_binomial_likelihood(d, int(1.0 * d), grid[x]) for x in range(g)]
    )
    d_1 = np.array(
        [log_binomial_likelihood(d, int(0.8 * d), grid[x]) for x in range(g)]
    )
    d_2 = np.array(
        [log_binomial_likelihood(d, int(0.6 * d), grid[x]) for x in range(g)]
    )
    d_3 = np.array(
        [log_binomial_likelihood(d, int(0.05 * d), grid[x]) for x in range(g)]
    )

    data_vals = [d_0, d_1, d_2, d_3]

elif data == "point_mass":
    ccfs = [5, 10, 15, 20, 100]
    # ccfs = [30, 70, 100]

    data_vals = []

    for c in ccfs:
        d = -1e100 * np.ones(101)

        d[c] = 0

        data_vals.append(d)

data = []

for i, d in enumerate(data_vals):
    data.append(DataPoint(i, np.atleast_2d(d)))

rng = instantiate_and_seed_RNG(None, None)

tree = Tree.get_single_node_tree(data)

conc_sampler = GammaPriorConcentrationSampler(0.01, 0.01, rng=rng)

# tree_dist = TreeJointDistribution(FSCRPDistribution(1.0))

tree_dist = TreeJointDistribution(UniformFSCRPDistribution(1.0))

mh_sampler = PruneRegraphSampler(tree_dist, rng=rng, labeled=True)

kernel = FullyAdaptedKernel(tree_dist, outlier_proposal_prob=0.0, rng=rng)

burnin_sampler = UnconditionalSMCSampler(
    kernel, num_particles=20, resample_threshold=0.5
)

pg_sampler = ParticleGibbsTreeSampler(
    kernel,
    num_particles=20,
    resample_threshold=0.5,
    rng=rng,
)

burnin = 100

num_iters = 1000

trace = []

for i in range(burnin):
    tree = burnin_sampler.sample_tree(tree)


for i in range(num_iters):
    tree = pg_sampler.sample_tree(tree)

    tree = mh_sampler.sample_tree(tree)

    trace.append(
        {
            "iter": i,
            "time": 0,
            "alpha": tree_dist.prior.alpha,
            "log_p": tree_dist.log_p_one(tree),
            "tree": tree.to_dict(),
        }
    )


topologies = []

for i, x in enumerate(trace):
    curr_tree = Tree.from_dict(data, x["tree"])
    count_topology(topologies, x, i, curr_tree)

df = _create_topology_dataframe(topologies)

df = df.sort_values(by="count", ascending=False)

print(df)