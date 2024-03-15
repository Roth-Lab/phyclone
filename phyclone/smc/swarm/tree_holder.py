from phyclone.tree import Tree


class TreeHolder(object):
    # __slots__ = 'log_w', 'parent_particle', 'tree', 'data', '_tree'

    def __init__(self, tree, tree_dist):

        # self.data = data

        self._tree_dist = tree_dist

        self.log_p = 0

        self.tree = tree

        self._hash_val = 0

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    def copy(self):
        return TreeHolder(self.tree, self._tree_dist)
        # TODO: re-write this? building tree unnecessarily here

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self.log_p = self._tree_dist.log_p(tree)
        # self.tree_roots = tree.roots
        self._data = tree.data
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()

    @tree.getter
    def tree(self):
        return Tree.from_dict(self._data, self._tree)
