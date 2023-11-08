from scipy.special import logsumexp as log_sum_exp

import numpy as np


class DataPoint(object):
    __slots__ = ('idx', 'value', 'name', 'outlier_prob', 'outlier_marginal_prob', 'outlier_prob_not')

    def __init__(self, idx, value, name=None, outlier_prob=0, outlier_prob_not=1):
        self.idx = idx

        self.value = value

        if name is None:
            name = idx

        self.name = name

        self.outlier_prob = outlier_prob

        self.outlier_prob_not = outlier_prob_not

        log_prior = -np.log(value.shape[1])

        self.outlier_marginal_prob = np.sum(log_sum_exp(self.value + log_prior, axis=1))

    @property
    def grid_size(self):
        return self.shape

    @property
    def shape(self):
        return self.value.shape
