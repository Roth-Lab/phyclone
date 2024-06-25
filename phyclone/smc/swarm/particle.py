from collections import deque

from phyclone.smc.swarm import TreeHolder


class Particle(object):
    __slots__ = (
        "_built_tree",
        "log_w",
        "parent_particle",
        "_tree_dist",
        "_perm_dist",
        "log_p",
        "log_pdf",
        "log_p_one",
        "_hash_val",
        "_tree",
        "tree_nodes",
        "tree_roots",
    )

    def __init__(self, log_w, parent_particle, tree_holder, tree_dist, perm_dist):
        self._built_tree = deque(maxlen=1)

        self.log_w = log_w

        self.parent_particle = parent_particle

        self._tree_dist = tree_dist

        self._perm_dist = perm_dist

        self.log_p = 0

        self.log_pdf = 0

        self.log_p_one = 0

        self._hash_val = 0

        self.tree = tree_holder

    def __hash__(self):
        return self._hash_val

    def __eq__(self, other):
        self_key = self._tree

        other_key = other._tree

        return self_key == other_key

    # def copy(self):
    #     cls = self.__class__
    #
    #     new = cls.__new__(cls)
    #
    #     new._built_tree = deque(maxlen=1)
    #     new.log_w = self.log_w
    #     new.parent_particle = self.parent_particle
    #     # new._data = self._data.copy()
    #     new._tree_dist = self._tree_dist
    #     new._perm_dist = self._perm_dist
    #     new.log_p = self.log_p
    #     new.log_pdf = self.log_pdf
    #     new.log_p_one = self.log_p_one
    #     new._hash_val = self._hash_val
    #     new.tree_roots = self.tree_roots.copy()
    #     new._tree = self._tree.copy()
    #     return new

    @property
    def tree(self):
        return self._tree

    @tree.getter
    def tree(self):
        return self._tree.tree

    @tree.setter
    def tree(self, tree):
        if not isinstance(tree, TreeHolder):
            tree = TreeHolder(tree, self._tree_dist, self._perm_dist)
        self.log_p = tree.log_p
        self.log_pdf = tree.log_pdf
        self.log_p_one = tree.log_p_one
        self.tree_roots = tree.tree_roots.copy()
        self.tree_nodes = tree.tree_nodes.copy()
        self._hash_val = hash(tree)
        self._tree = tree

    @property
    def built_tree(self):
        return self._built_tree

    @built_tree.setter
    def built_tree(self, tree):
        self._built_tree.append(tree)

    @built_tree.getter
    def built_tree(self):
        return self._built_tree.pop()
