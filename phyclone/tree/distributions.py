import numpy as np

from phyclone.utils.math import log_sum_exp, cached_log_factorial


class FSCRPDistribution(object):
    """FSCRP prior distribution on trees."""

    __slots__ = ("_alpha", "log_alpha", "_c_const")

    def __init__(self, alpha, c_const=1000):
        self.alpha = alpha
        self.c_const = c_const

    def __eq__(self, other):
        alpha_check = self.alpha == other.alpha
        return alpha_check

    def __hash__(self):
        return hash(self.alpha)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self.log_alpha = np.log(alpha)

    @property
    def c_const(self):
        return self._c_const

    @c_const.setter
    def c_const(self, c_const):
        self._c_const = np.log(c_const)

    def log_p(
        self, tree, tree_node_data=None, log_p=None, num_nodes=None, multiplicity=None
    ):
        if not log_p or not num_nodes:
            log_p, num_nodes = self._alpha_and_CRP_prior_log_p_compute(
                tree, tree_node_data
            )

        if not multiplicity:
            multiplicity = tree.multiplicity

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        log_p -= multiplicity

        return log_p

    def _alpha_and_CRP_prior_log_p_compute(self, tree, tree_node_data):
        if tree_node_data is None:
            tree_node_data = tree.node_data
        log_p = 0
        # CRP prior
        num_nodes = tree.get_number_of_nodes()
        log_p += num_nodes * self.log_alpha
        log_p += sum(
            cached_log_factorial(len(v) - 1)
            for k, v in tree_node_data.items()
            if k != -1
        )
        return log_p, num_nodes

    def log_p_one(
        self, tree, tree_node_data=None, log_p=None, num_nodes=None, multiplicity=None
    ):
        if not log_p or not num_nodes:
            log_p, num_nodes = self._alpha_and_CRP_prior_log_p_compute(
                tree, tree_node_data
            )

        if not multiplicity:
            multiplicity = tree.multiplicity

        tree_roots = tree.roots

        num_ways = 0

        r_term = self._compute_r_term(len(tree_roots), num_nodes)

        for root in tree_roots:
            curr_num_nodes = tree.get_number_of_descendants(root) + 1
            num_sub_trees = (curr_num_nodes - 1) * np.log(curr_num_nodes)
            num_ways += num_sub_trees

        log_p += -num_ways + r_term

        log_p -= multiplicity

        return log_p

    def compute_both_log_p_and_log_p_one_priors(self, tree, tree_node_data=None):
        log_p_start, num_nodes = self._alpha_and_CRP_prior_log_p_compute(
            tree, tree_node_data
        )

        multiplicity = tree.multiplicity

        log_p_one = log_p_start
        log_p = log_p_start

        log_p = self.log_p(tree, tree_node_data, log_p, num_nodes, multiplicity)

        log_p_one = self.log_p_one(
            tree, tree_node_data, log_p_one, num_nodes, multiplicity
        )

        return log_p, log_p_one

    def _compute_z_term(self, num_roots, num_nodes):
        # TODO: this is just 0, any point to doing this?
        log_one = np.log(1)

        # TODO: 1 raised to the power of anything is still just 1, likely don't need this
        a_term = (log_one * num_nodes)

        la = log_one
        log_const = self.c_const

        if num_roots == 0:
            # r_term_denominator = log_one - (log_const * 1)
            # r_term_denominator = la + np.log1p(-np.exp(r_term_denominator - la))
            # res = a_term - r_term_denominator
            res = a_term
        else:

            r_term_numerator = log_one - (log_const * num_roots)
            r_term_denominator = log_one - (log_const * 1)

            r_term_numerator = la + np.log1p(-np.exp(r_term_numerator - la))
            r_term_denominator = la + np.log1p(-np.exp(r_term_denominator - la))

            res = a_term + (r_term_numerator - r_term_denominator)
        return res

    def _compute_r_term(self, num_roots, num_nodes):
        z_term = self._compute_z_term(num_roots, num_nodes)
        log_const = self.c_const

        if num_roots == 0:
            num_roots = 1

        return np.log(1) - (z_term + (log_const * (num_roots - 1)))


class TreeJointDistribution(object):
    __slots__ = "prior"

    def __init__(self, prior):
        self.prior = prior

    def __eq__(self, other):
        return self.prior == other.prior

    def __hash__(self):
        return hash(self.prior)

    def log_p(self, tree):
        """The log likelihood of the data marginalized over root node parameters."""

        tree_node_data = tree.node_data

        log_p = self.prior.log_p(tree, tree_node_data)

        # Outlier prior
        log_p += self.outlier_prior(tree_node_data)

        if tree.get_number_of_children("root") > 0:
            for i in range(tree.grid_size[0]):
                log_p += log_sum_exp(tree.data_log_likelihood[i, :])

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p

    def log_p_one(self, tree):
        """The log likelihood of the data conditioned on the root having value 1.0 in all dimensions."""

        tree_node_data = tree.node_data

        log_p = self.prior.log_p_one(tree, tree_node_data)

        # Outlier prior
        log_p += self.outlier_prior(tree_node_data)

        if tree.get_number_of_children("root") > 0:
            for i in range(tree.grid_size[0]):
                log_p += tree.data_log_likelihood[i, -1]

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p

    def compute_both_log_p_and_log_p_one(self, tree):
        tree_node_data = tree.node_data

        log_p, log_p_one = self.prior.compute_both_log_p_and_log_p_one_priors(
            tree, tree_node_data
        )

        log_outlier_prior = self.outlier_prior(tree_node_data)

        log_p += log_outlier_prior

        log_p_one += log_outlier_prior

        if tree.get_number_of_children("root") > 0:
            for i in range(tree.grid_size[0]):
                log_p += log_sum_exp(tree.data_log_likelihood[i, :])
                log_p_one += tree.data_log_likelihood[i, -1]

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob
            log_p_one += data_point.outlier_marginal_prob

        return log_p, log_p_one

    @staticmethod
    def outlier_prior(tree_node_data):
        log_p = 0
        for node, node_data in tree_node_data.items():
            for data_point in node_data:
                if data_point.outlier_prob != 0:
                    if node == -1:
                        log_p += data_point.outlier_prob

                    else:
                        log_p += data_point.outlier_prob_not
        return log_p
