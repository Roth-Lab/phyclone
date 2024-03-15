from collections import deque

from phyclone.tree import Tree


class Particle(object):
    # __slots__ = 'log_w', 'parent_particle', 'tree', 'data', '_tree'

    def __init__(self, log_w, parent_particle, tree, tree_dist, perm_dist):
        self._built_tree = deque(maxlen=1)

        self.log_w = log_w

        self.parent_particle = parent_particle

        # self.data = data

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        self.log_p = 0

        self.log_pdf = 0

        self.log_p_one = 0

        self.tree = tree

        self._hash_val = 0

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    def copy(self):
        cls = self.__class__

        new = cls.__new__(cls)

        new._built_tree = deque(maxlen=1)
        new.log_w = self.log_w
        new.parent_particle = self.parent_particle
        new._data = self._data.copy()
        new._tree_dist = self._tree_dist
        new._perm_dist = self._perm_dist
        new.log_p = self.log_p
        new.log_pdf = self.log_pdf
        new.log_p_one = self.log_p_one
        new._hash_val = self._hash_val
        new.tree_roots = self.tree_roots.copy()
        new._tree = self._tree.copy()
        return new
        # return Particle(self.log_w, self.parent_particle, self.tree, self.data, self._tree_dist, self._perm_dist)

    @property
    def tree(self):
        return self._tree

    @tree.getter
    def tree(self):
        return Tree.from_dict(self._data, self._tree)
        # return self._tree.copy()

    @tree.setter
    def tree(self, tree):
        self._data = tree.data
        self.log_p = self._tree_dist.log_p(tree)
        if self._perm_dist is None:
            self.log_pdf = 0.0
        else:
            self.log_pdf = self._perm_dist.log_pdf(tree)
        self.log_p_one = self._tree_dist.log_p_one(tree)
        self.tree_roots = tree.roots
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()
        # self._tree = tree

    @property
    def built_tree(self):
        return self._built_tree

    @built_tree.setter
    def built_tree(self, tree):
        self._built_tree.append(tree)

    @built_tree.getter
    def built_tree(self):
        return self._built_tree.pop()
