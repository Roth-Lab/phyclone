import numpy as np

from phyclone.math_utils import log_factorial, log_sum_exp


class FSCRPDistribution(object):
    """FSCRP prior distribution on trees."""

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


class TreeJointDistribution(object):
    def __init__(self, prior):
        self.prior = prior

    def log_p(self, tree):
        """The log likelihood of the data marginalized over root node parameters."""
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
        """The log likelihood of the data conditioned on the root having value 1.0 in all dimensions."""
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
