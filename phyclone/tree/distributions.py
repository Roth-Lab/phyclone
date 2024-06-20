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

    def log_p(self, tree, tree_node_data=None):
        if tree_node_data is None:
            tree_node_data = tree.node_data

        log_p = 0

        # CRP prior
        num_nodes = tree.get_number_of_nodes()

        log_p += num_nodes * self.log_alpha

        log_p += sum(cached_log_factorial(len(v) - 1) for k, v in tree_node_data.items() if k != -1)

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        log_p -= tree.multiplicity

        return log_p

    def log_p_one(self, tree, tree_node_data=None):
        if tree_node_data is None:
            tree_node_data = tree.node_data

        log_p = 0

        # CRP prior
        num_nodes = tree.get_number_of_nodes()

        log_p += num_nodes * self.log_alpha

        log_p += sum(cached_log_factorial(len(v) - 1) for k, v in tree_node_data.items() if k != -1)

        tree_roots = tree.roots

        num_ways = 0

        r_term = self._compute_r_term(len(tree_roots), num_nodes)

        for root in tree_roots:
            num_nodes = tree.get_number_of_descendants(root) + 1
            num_sub_trees = (num_nodes - 1) * np.log(num_nodes)
            num_ways += num_sub_trees

        log_p += (-num_ways + r_term)

        log_p -= tree.multiplicity

        return log_p

    def _compute_z_term(self, num_roots, num_nodes):
        log_one = np.log(1)  # TODO: this is just 0, any point to doing this?
        a_term = log_one * num_nodes  # TODO: 1 raised to the power of anything is still just 1, likely don't need this
        la = log_one

        log_const = self.c_const

        if num_roots == 0:
            r_term_denominator = log_one - (log_const * 1)
            r_term_denominator = la + np.log1p(-np.exp(r_term_denominator - la))
            res = a_term - r_term_denominator
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

        # log_p = self.prior.log_p(tree, tree_node_data)

        log_p = self.prior.log_p_one(tree, tree_node_data)

        # Outlier prior
        log_p += self.outlier_prior(tree_node_data)

        if tree.get_number_of_children("root") > 0:
            for i in range(tree.grid_size[0]):
                log_p += tree.data_log_likelihood[i, -1]

        for data_point in tree.outliers:
            log_p += data_point.outlier_marginal_prob

        return log_p

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
