class DataPoint(object):
    __slots__ = ('idx', 'value')

    def __init__(self, idx, value):
        self.idx = idx

        self.value = value

    @property
    def shape(self):
        return self.value.shape
