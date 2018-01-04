import numpy as np

from phyclone.math_utils import log_normalize, log_sum_exp


class DataPoint(object):
    __slots__ = ('idx', 'value', 'outlier_prob')

    def __init__(self, idx, value, outlier_prob=0):
        self.idx = idx

        self.value = np.zeros(value.shape)

        for i in range(value.shape[0]):
            self.value[i, :] = log_normalize(value[i, :])

            assert log_sum_exp(self.value[i]) == 0

        self.outlier_prob = outlier_prob

    @property
    def grid_size(self):
        return self.shape

    @property
    def shape(self):
        return self.value.shape
