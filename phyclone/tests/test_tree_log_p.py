import unittest

import numpy as np

from phyclone.tree import FSCRPDistribution, Tree, TreeJointDistribution
from phyclone.math_utils import log_factorial, log_sum_exp
from phyclone.tests.simulate import simulate_binomial_data


class OldFSCRPDistribution(object):
    """ FSCRP prior distribution on trees.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, tree):
        log_p = 0

        # CRP prior
        num_nodes = len(tree.nodes)

        log_p += num_nodes * np.log(self.alpha)

        for node, node_data in tree.node_data.items():
            if node == -1:
                continue

            num_data_points = len(node_data)

            log_p += log_factorial(num_data_points - 1)

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        return log_p


class OldTreeJointDistribution(object):

    def __init__(self, prior):
        self.prior = prior

    def log_p(self, tree):
        """ The log likelihood of the data marginalized over root node parameters.
        """
        log_p = self.prior.log_p(tree)

        # Outlier prior
        for node, node_data in tree.node_data.items():
            for data_point in node_data:
                if data_point.outlier_prob != 0:
                    if node == -1:
                        log_p += data_point.outlier_prob

                    else:
                        log_p += data_point.outlier_prob_not

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += log_sum_exp(tree.data_log_likelihood[i, :])

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p

    def log_p_one(self, tree):
        """ The log likelihood of the data conditioned on the root having value 1.0 in all dimensions.
        """
        log_p = self.prior.log_p(tree)

        # Outlier prior
        for node, node_data in tree.node_data.items():
            for data_point in node_data:
                if data_point.outlier_prob != 0:
                    if node == -1:
                        log_p += data_point.outlier_prob

                    else:
                        log_p += data_point.outlier_prob_not

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += tree.data_log_likelihood[i, -1]

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p


class Test(unittest.TestCase):

    def setUp(self):

        self.outlier_prob = 0.0

        self._rng = np.random.default_rng(12345)

        self.tree_dist = FSCRPDistribution(1.0)

        self.tree_joint_dist = TreeJointDistribution(self.tree_dist)

        self.expected_tree_dist = OldFSCRPDistribution(1.0)

        self.expected_tree_joint_dist = OldTreeJointDistribution(self.expected_tree_dist)

    def test_build_single_node_tree_1DP_1clust_1D_0_outlier_prob(self):
        n = 100
        p = 1.0

        data = self._create_data_points(1, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_1DP_1clust_1D_1pct_outlier_prob(self):
        n = 100
        p = 1.0

        self.outlier_prob = 0.01

        data = self._create_data_points(1, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_1clust_1D_0_outlier_prob(self):
        n = 100
        p = 1.0

        data = self._create_data_points(2, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_1clust_1D_1pct_outlier_prob(self):
        n = 100
        p = 1.0

        self.outlier_prob = 0.01

        data = self._create_data_points(2, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_1DP_1clust_2D_0_outlier_prob(self):
        n = 100
        p = [1.0, 1.0]

        data = self._create_data_points(1, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_1DP_1clust_2D_1pct_outlier_prob(self):
        n = 100
        p = [1.0, 1.0]

        self.outlier_prob = 0.01

        data = self._create_data_points(1, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_1clust_2D_0_outlier_prob(self):
        n = 100
        p = [1.0, 1.0]

        data = self._create_data_points(2, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_1clust_2D_1pct_outlier_prob(self):
        n = 100
        p = [1.0, 1.0]

        self.outlier_prob = 0.01

        data = self._create_data_points(2, n, p)

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_2clust_1D_0_outlier_prob(self):
        n = 100
        c1_p = 1.0
        c2_p = 0.5

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_2clust_1D_1pct_outlier_prob(self):
        self.outlier_prob = 0.01

        n = 100
        c1_p = 1.0
        c2_p = 0.5

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_2clust_2D_0_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_build_single_node_tree_2DP_2clust_2D_1pct_outlier_prob(self):
        self.outlier_prob = 0.01

        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        self._run_asserts(tree)

    def test_add_1DP_single_node_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        new_datapoint = self._create_data_point(len(data), n, c1_p)

        tree.add_data_point_to_node(new_datapoint, 0)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_add_1DP_single_node_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01

        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        new_datapoint = self._create_data_point(len(data), n, c1_p)

        tree.add_data_point_to_node(new_datapoint, 0)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_remove_1DP_single_node_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        tree.remove_data_point_from_node(clust_1[0], 0)

        assert len(tree.labels) == 1

        self._run_asserts(tree)

    def test_remove_1DP_single_node_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01

        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        clust_1 = self._create_data_points(1, n, c1_p)
        clust_2 = self._create_data_points(1, n, c2_p, start_idx=len(clust_1))

        data = clust_1 + clust_2

        tree = Tree.get_single_node_tree(data)

        tree.remove_data_point_from_node(clust_1[0], 0)

        assert len(tree.labels) == 1

        self._run_asserts(tree)

    def test_add_1DP_cherry_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        assert len(tree.labels) == 4

        self._run_asserts(tree)

    def test_add_1DP_cherry_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        assert len(tree.labels) == 4

        self._run_asserts(tree)

    def test_remove_1DP_cherry_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)
        tree.remove_data_point_from_node(dp_2[0], n_1)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_remove_1DP_cherry_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        tree.remove_data_point_from_node(dp_2[0], n_1)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_copy_cherry_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        assert len(tree.labels) == 4

        copied_tree = tree.copy()
        assert tree == copied_tree
        self._run_asserts(tree)
        self._run_asserts(copied_tree)

    def test_copy_cherry_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        assert len(tree.labels) == 4

        copied_tree = tree.copy()
        assert tree == copied_tree
        self._run_asserts(tree)
        self._run_asserts(copied_tree)

    def test_copy_subtree_true_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        sub_tree = tree.get_subtree(n_2)
        assert tree == sub_tree
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_copy_subtree_not_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, n_1)

        sub_tree = tree.get_subtree(n_1)
        assert tree != sub_tree
        assert len(tree.nodes) == 3
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_add_1DP_to_outliers_cherry_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, -1)

        assert len(tree.labels) == 4

        self._run_asserts(tree)

    def test_add_1DP_to_outliers_cherry_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, -1)

        assert len(tree.labels) == 4

        self._run_asserts(tree)

    def test_remove_1DP_to_outliers_cherry_tree_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, -1)

        assert len(tree.labels) == 4

        tree.remove_data_point_from_node(dp_4, -1)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_remove_1DP_to_outliers_cherry_tree_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        tree.add_data_point_to_node(dp_4, -1)

        assert len(tree.labels) == 4

        tree.remove_data_point_from_node(dp_4, -1)

        assert len(tree.labels) == 3

        self._run_asserts(tree)

    def test_add_subtree_not_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 4
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_add_subtree_not_root_cherry_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 4
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_add_subtree_2_nodes_not_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)
        dp_5 = self._create_data_point(4, n, c1_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        sn_1 = sub_tree.create_root_node(children=[0], data=[dp_5])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 5
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_add_subtree_2_nodes_not_root_cherry_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)
        dp_5 = self._create_data_point(4, n, c1_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        sn_1 = sub_tree.create_root_node(children=[0], data=[dp_5])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 5
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_remove_subtree_2_nodes_not_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)
        dp_5 = self._create_data_point(4, n, c1_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        sn_1 = sub_tree.create_root_node(children=[0], data=[dp_5])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 5
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

        tree.remove_subtree(sub_tree)

        assert tree != sub_tree
        assert len(tree.nodes) == 3
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_remove_subtree_2_nodes_not_root_cherry_0pct_outlier_prob(self):
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)
        dp_5 = self._create_data_point(4, n, c1_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        sn_1 = sub_tree.create_root_node(children=[0], data=[dp_5])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 5
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

        tree.remove_subtree(sub_tree)

        assert tree != sub_tree
        assert len(tree.nodes) == 3
        assert len(sub_tree.nodes) == 2
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_remove_subtree_not_root_cherry_1pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 4
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

        tree.remove_subtree(sub_tree)

        assert tree != sub_tree
        assert len(tree.nodes) == 3
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def test_remove_subtree_not_root_cherry_0pct_outlier_prob(self):
        self.outlier_prob = 0.01
        n = 100
        c1_p = [1.0, 1.0]
        c2_p = [0.5, 0.7]

        root_dp = self._create_data_points(1, n, c1_p)
        dp_2 = self._create_data_points(1, n, c2_p, start_idx=len(root_dp))

        tree = Tree.get_single_node_tree(root_dp)

        dp_3 = self._create_data_point(2, n, c1_p)
        dp_4 = self._create_data_point(3, n, c2_p)

        n_1 = tree.create_root_node(children=[], data=dp_2)

        n_2 = tree.create_root_node(children=[n_1, 0], data=[dp_3])

        sub_tree = Tree.get_single_node_tree([dp_4])

        tree.add_subtree(sub_tree, n_1)

        assert tree != sub_tree
        assert len(tree.nodes) == 4
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

        tree.remove_subtree(sub_tree)

        assert tree != sub_tree
        assert len(tree.nodes) == 3
        assert len(sub_tree.nodes) == 1
        self._run_asserts(tree)
        self._run_asserts(sub_tree)

    def _run_asserts(self, tree):
        expected_log_p_one = self.expected_tree_joint_dist.log_p_one(tree)
        expected_log_p_FSCRP = self.expected_tree_dist.log_p(tree)
        expected_log_p = self.expected_tree_joint_dist.log_p(tree)
        actual_log_p_one = self.tree_joint_dist.log_p_one(tree)
        actual_log_p_FSCRP = self.tree_dist.log_p(tree)
        actual_log_p = self.tree_joint_dist.log_p(tree)

        np.testing.assert_allclose(actual_log_p_FSCRP, expected_log_p_FSCRP)
        np.testing.assert_allclose(actual_log_p, expected_log_p)
        np.testing.assert_allclose(actual_log_p_one, expected_log_p_one)

    def _create_data_point(self, idx, n, p):
        return simulate_binomial_data(idx, n, p, self._rng, self.outlier_prob)

    def _create_data_points(self, size, n, p, start_idx=0):
        result = []

        for i in range(size):
            result.append(self._create_data_point(i+start_idx, n, p))

        return result
