class DataPoint(object):
    __slots__ = ('idx', 'value', 'outlier_prob')

    def __init__(self, idx, value, outlier_prob=0):
        self.idx = idx

        self.value = value

        self.outlier_prob = outlier_prob

    @property
    def grid_size(self):
        return self.shape

    @property
    def shape(self):
        return self.value.shape
