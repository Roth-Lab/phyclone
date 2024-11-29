import unittest
import numpy as np
from dataclasses import dataclass
from fscrp import simulate_fscrp_tree
from phyclone.tests.data import simulate_data
from phyclone.tree import Tree
import networkx as nx
from scipy.stats import dirichlet
from phyclone.utils.math import log_factorial
from importance_sampler import run_importance_sampler
from scipy.stats import ttest_ind


@dataclass
class DataParams:
    alpha: float
    depth: int
    max_cn: int
    min_cn: int
    min_minor_cn: int
    num_samples: int
    num_snvs: int
    tumour_content: float
    density: str
    precision: float
    grid_size: int
    num_trials: int


def build_phyclone_tree_from_nx(nx_tree):
    grid_size = nx_tree.graph["shape"]
    tree_root = nx_tree.graph["root"]

    phyclone_tree = Tree(grid_size)

    post_order_nodes = nx.dfs_postorder_nodes(nx_tree, tree_root)

    node_map = {}

    for node in post_order_nodes:

        children = list(nx_tree.successors(node))
        translated_children = []
        for child in children:
            translated_children.append(node_map[child])

        data = nx_tree.nodes[node]["datapoint"]
        new_node = phyclone_tree.create_root_node(translated_children, data)
        node_map[node] = new_node

    return phyclone_tree


def get_node_post_order(tree):
    root = tree.graph["root"]
    node_post_order = list(nx.dfs_postorder_nodes(tree, root))
    return node_post_order


def establish_child_lists_on_nodes(node_post_order, tree):
    node_post_order_stripped = node_post_order.copy()
    for node in node_post_order:
        children = list(tree.successors(node))
        if len(children) == 0:
            node_post_order_stripped.remove(node)
        tree.nodes[node]["children"] = np.array(children, dtype=np.int64)

    tree.graph["node_post_order_stripped"] = node_post_order_stripped


def compute_tree_prior_term(alpha, num_nodes, tree):
    log_alpha = np.log(alpha)
    num_nodes += 1
    log_p_prior = num_nodes + log_alpha

    for node in tree.nodes:
        num_snvs = len(tree.nodes[node]["snvs"])
        log_p_prior += log_factorial(num_snvs - 1)

    log_p_prior -= (num_nodes - 1) * np.log(num_nodes + 1)

    return log_p_prior


class Test(unittest.TestCase):

    def __init__(self, method_name: str = ...):
        super().__init__(method_name)

        self.grid_size = 101
        self.rng = np.random.default_rng(12345)
        self.num_iters = 100000
        self.pval_threshold = 0.001

    def simulate_fscrp_tree(self, data_params):
        tree = simulate_fscrp_tree(
            self.rng, alpha=data_params.alpha, dim=data_params.num_samples, num_data_points=data_params.num_snvs
        )
        return tree

    def simulate_data_on_tree(self, tree, data_params):
        simulate_data(
            tree,
            self.rng,
            data_params.num_samples,
            depth=data_params.depth,
            max_cn=data_params.max_cn,
            min_cn=data_params.min_cn,
            min_minor_cn=data_params.min_minor_cn,
            tumour_content=data_params.tumour_content,
            error_rate=1e-3,
        )
        grid_size = data_params.grid_size
        num_nodes = tree.number_of_nodes()
        tree.graph["num_samples"] = data_params.num_samples
        tree.graph["num_nodes"] = num_nodes
        tree.graph["shape"] = (data_params.num_samples, grid_size)
        tree.graph["num_snvs"] = data_params.num_snvs
        node_post_order = get_node_post_order(tree)
        establish_child_lists_on_nodes(node_post_order, tree)

        self.node_post_order = node_post_order

        self.log_prior = compute_tree_prior_term(data_params.alpha, num_nodes, tree)

        self.sim_tree = tree

        ones_arr = np.ones(num_nodes)
        diri_prior = dirichlet(ones_arr, seed=self.rng)
        self.diri_prior_val = diri_prior.logpdf(diri_prior.rvs().T)
        self.diri_dist = diri_prior

    def get_importance_sampler_likelihood(self, data_params, trial):
        importance_sampler_likelihood = run_importance_sampler(
            self.num_iters,
            self.sim_tree,
            self.diri_dist,
            data_params.density,
            data_params.precision,
            self.node_post_order,
            self.log_prior,
            trial + 1,
        )
        return importance_sampler_likelihood

    def _run_test(self, data_params):

        phyclone_llh_arr = np.empty(data_params.num_trials, dtype=np.float64)

        importance_sampler_llh_arr = np.empty(data_params.num_trials, dtype=np.float64)

        for i in range(data_params.num_trials):
            tree = self.simulate_fscrp_tree(data_params)

            self.simulate_data_on_tree(tree, data_params)

            phyclone_tree = build_phyclone_tree_from_nx(tree)

            phyclone_llh = self.compute_phyclone_log_p(phyclone_tree)

            importance_sampler_likelihood = self.get_importance_sampler_likelihood(data_params, i)

            phyclone_llh_arr[i] = phyclone_llh
            importance_sampler_llh_arr[i] = importance_sampler_likelihood

        ttest_ind_result = ttest_ind(phyclone_llh_arr, importance_sampler_llh_arr, random_state=self.rng)

        print("IS likelihoods: \n{}".format(importance_sampler_llh_arr))
        print("PhyClone likelihoods: \n{}".format(phyclone_llh_arr))
        print("\nT-test result: \n{}".format(ttest_ind_result))

        assert ttest_ind_result.pvalue > self.pval_threshold

    def compute_phyclone_log_p(self, tree):
        grid_size = self.grid_size

        log_p = 0

        log_p += self.log_prior

        for i in range(tree.grid_size[0]):
            log_p += tree.data_log_likelihood[i, -1]

        grid_prior = np.log(grid_size) * 2

        log_p += grid_prior

        log_p += self.diri_prior_val

        return log_p

    def test_50_snvs(self):
        data_params = DataParams(
            alpha=1.0,
            depth=1000,
            max_cn=2,
            min_cn=2,
            min_minor_cn=1,
            num_samples=1,
            num_snvs=50,
            tumour_content=1.0,
            density="beta-binomial",
            precision=400,
            grid_size=self.grid_size,
            num_trials=10,
        )

        self._run_test(data_params)

    def test_100_snvs(self):
        data_params = DataParams(
            alpha=1.0,
            depth=1000,
            max_cn=2,
            min_cn=2,
            min_minor_cn=1,
            num_samples=1,
            num_snvs=100,
            tumour_content=1.0,
            density="beta-binomial",
            precision=400,
            grid_size=self.grid_size,
            num_trials=10,
        )

        self._run_test(data_params)

    def test_100_snvs_2_samples(self):
        data_params = DataParams(
            alpha=1.0,
            depth=1000,
            max_cn=2,
            min_cn=2,
            min_minor_cn=1,
            num_samples=2,
            num_snvs=100,
            tumour_content=1.0,
            density="beta-binomial",
            precision=400,
            grid_size=self.grid_size,
            num_trials=10,
        )

        self._run_test(data_params)

    def test_1000_snvs(self):
        data_params = DataParams(
            alpha=1.0,
            depth=1000,
            max_cn=2,
            min_cn=2,
            min_minor_cn=1,
            num_samples=1,
            num_snvs=1000,
            tumour_content=1.0,
            density="beta-binomial",
            precision=400,
            grid_size=self.grid_size,
            num_trials=10,
        )

        self._run_test(data_params)


if __name__ == "__main__":
    unittest.main()
