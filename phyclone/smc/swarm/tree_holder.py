from phyclone.tree import Tree


class TreeHolder(object):
    __slots__ = ('_tree_dist', 'log_p', '_hash_val', '_tree', 'log_pdf', 'log_p_one', '_perm_dist',
                 "tree_nodes", "tree_roots", "labels")

    def __init__(self, tree, tree_dist, perm_dist):

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        self.log_p = 0

        self.log_pdf = 0

        self.log_p_one = 0

        self._hash_val = 0

        self.tree = tree

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    def copy(self):
        return TreeHolder(self.tree, self._tree_dist, self._perm_dist)
        # TODO: re-write this? building tree unnecessarily here

    @property
    def tree(self):
        return self._tree

    # @tree.setter
    # def tree(self, tree):
    #     self.log_p = self._tree_dist.log_p(tree)
    #     # self._data = tree.data
    #     self._hash_val = hash(tree)
    #     self._tree = tree.to_dict()
    @tree.setter
    def tree(self, tree):
        self.log_p = self._tree_dist.log_p(tree)
        if self._perm_dist is None:
            self.log_pdf = 0.0
        else:
            self.log_pdf = self._perm_dist.log_pdf(tree)
        self.log_p_one = self._tree_dist.log_p_one(tree)
        self.tree_roots = tree.roots
        self.tree_nodes = tree.nodes
        self._hash_val = hash(tree)
        self._tree = tree.to_dict()
        self.labels = tree.labels

    @tree.getter
    def tree(self):
        return Tree.from_dict(self._tree)
