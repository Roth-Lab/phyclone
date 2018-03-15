import gzip
import pickle

import phyclone.tree


class Trace(object):
    def __init__(self, file_name, mode='r'):
        self.mode = mode

        self._fh = gzip.GzipFile(file_name, mode='{}b'.format(mode))

    def close(self):
        self._fh.close()

    def load(self, data):
        assert self.mode == 'r'

        trees = []

        while True:
            try:
                tree_dict = pickle.load(self._fh)

                trees.append(
                    phyclone.tree.Tree.from_dict(data, tree_dict)
                )

            except EOFError:
                break

        return trees

    def update(self, tree):
        assert self.mode == 'w'

        pickle.dump(tree.to_dict(), self._fh)
