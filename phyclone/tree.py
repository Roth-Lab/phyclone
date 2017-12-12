'''
Created on 16 Sep 2017

@author: Andrew Roth
'''
from scipy.signal import fftconvolve

import networkx as nx
import numpy as np

from phyclone.math_utils import log_sum_exp


class Tree(object):
    """ FSCRP tree data structure.

    This structure includes the dummy root node.
    """

    def __init__(self, nodes):
        """
        Parameters
        ----------
        data_points: dict
            A mapping from nodes to a list of data points associated with the node.
        nodes: list
            A list of MarginalNodes
        """

#         self._nodes = {}
#
#         for n in nodes:
#             self._nodes[n.idx] = n

        self._init_graph(nodes)

        self._validate()

    def _init_graph(self, nodes):
        # Create graph
        G = nx.DiGraph()

        for node in nodes:
            G.add_node(node.idx)

            for child in node.children:
                G.add_edge(node.idx, child.idx)

        # Connect roots to dummy root
        G.add_node(-1)

        for node in nodes:
            if G.in_degree(node.idx) == 0:
                G.add_edge(-1, node.idx)

        self._graph = G

        # Instantiate dummy node
        roots = []

        for node in nodes:
            if node.idx in self._graph.successors(-1):
                roots.append(node)

        dummy_root = MarginalNode(
            -1,
            nodes[0].grid_size,
            children=roots
        )

        self._nodes = {}

        for n in get_nodes(dummy_root):
            self._nodes[n.idx] = n

    @property
    def data_points(self):
        return set(self.labels.keys())

    @property
    def nodes(self):
        nodes = self._nodes.copy()

        del nodes[-1]

        return nodes

    @property
    def graph(self):
        graph = self._graph.copy()

        graph.remove_node(-1)

        return graph

    @property
    def log_p(self):
        return self._nodes[-1].log_p

    @property
    def log_p_one(self):
        return self._nodes[-1].log_p_one

    @property
    def labels(self):
        labels = {}

        for node in self.nodes.values():
            for data_point in node.data:
                labels[data_point.idx] = node.idx

        return labels

    @property
    def roots(self):
        return [self._nodes[idx] for idx in self._graph.successors(-1)]

    def copy(self):
        root = self._nodes[-1].copy()

        return Tree(get_nodes(root)[1:])

    def draw(self, ax=None):
        nx.draw(
            self.graph,
            ax=ax,
            pos=nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot'),
            with_labels=True
        )

    def get_children_nodes(self, node):
        return node.children

    def get_parent_node(self, node):
        if node.idx == -1:
            return None

        parent_idxs = list(self._graph.predecessors(node.idx))

        assert len(parent_idxs) == 1

        parent_idx = parent_idxs[0]

        return self._nodes[parent_idx]

#     def add_node(self, data_points, node, children=None, parent=None):
#         if node.idx in self._nodes:
#             raise Exception('Node {} exists in tree'.format(node.idx))
#
#         self._data_points[node.idx] = data_points
#
#         # Node editing
#         self._nodes[node.idx] = node
#
#         if children is not None:
#             node.update_children(children)
#
#             for child in children:
#                 self._graph.add_edge(node.idx, child.idx)
#
#         if parent is not None:
#             parent.add_child_node(node)
#
#             self._update_ancestor_nodes(parent)
#
#             self._graph.add_edge(parent.idx, node.idx)
#
#         self.get_parent_node(node)
#
#     def remove_node(self, node):
#         if node.idx == -1:
#             raise Exception('Cannot remove root node')
#
#         # Node editing
#         parent = self.get_parent_node(node)
#
#         parent.remove_child_node(node)
#
#         self._update_ancestor_nodes(parent)
#
#         del self._data_points[node.idx]
#
#         del self._nodes[node.idx]
#
#         # Graph editing
#         for edge in self._graph.edges():
#             if node.idx in edge:
#                 self._graph.remove_edge(*edge)
#
#         self._graph.remove_node(node.idx)

    def add_subtree(self, subtree, parent=None):
        if parent is None:
            parent = self._nodes[-1]

        assert parent in self._nodes.values()

        assert len(subtree.data_points & self.data_points) == 0

        for n in subtree.nodes.values():
            assert n.idx not in self._nodes.keys()

            assert n not in self._nodes.values()

            self._nodes[n.idx] = n

            for child in n.children:
                self._graph.add_edge(n.idx, child.idx)

        for node in subtree.roots:
            self._graph.add_edge(parent.idx, node.idx)

            parent.add_child_node(node)

            assert parent == self.get_parent_node(node)

            assert node in parent.children

        self._update_ancestor_nodes(parent)

        self._validate()

    def get_subtree(self, subtree_root):
        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        nodes = []

        for node_idx in subtree_node_idxs:
            nodes.append(self._nodes[node_idx])

        t = Tree(nodes)

        t._validate()

        return t

    def remove_subtree(self, tree):
        for root in tree.roots:
            subtree = tree.get_subtree(root)

            self._remove_subtree(subtree)

    def relabel_nodes(self, min_value=0):
        # TODO: Fix this by copy the dicts
        node_map = {}

        self._data_points = {-1: []}

        self._nodes = {-1: self._nodes[-1].copy()}

        old_nodes = get_nodes(self._nodes[-1])

        for new_idx, old in enumerate(old_nodes[1:], min_value):
            node_map[old.idx] = new_idx

            self._nodes[new_idx] = old

            self._nodes[new_idx].idx = new_idx

        self._graph = nx.relabel_nodes(self._graph, node_map)

        for node_idx in nx.dfs_preorder_nodes(self._graph, -1):
            node = self._nodes[node_idx]

            node._children = {}

            for child_idx in self._graph.successors(node_idx):
                child = self._nodes[child_idx]

                node._children[child_idx] = child

                assert child_idx == child.idx

        self._validate()

    def _remove_subtree(self, subtree):
        assert len(subtree.roots) == 1

        subtree_root = subtree.roots[0]

        parent = self.get_parent_node(subtree_root)

        parent.remove_child_node(subtree_root)

        assert subtree_root not in parent.children

        self._update_ancestor_nodes(parent)

        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        # Update data structures
        for node_idx in subtree_node_idxs:
            del self._nodes[node_idx]

            self._graph.remove_node(node_idx)

        self._validate()

    def _update_ancestor_nodes(self, source):
        '''
        Update all ancestor _nodes of source sequentially from source to root.
        '''
        while source is not None:
            self._nodes[source.idx].update()

            source = self.get_parent_node(source)

    def _validate(self):
        for node in self._nodes.values():
            assert set(self._graph.successors(node.idx)) == set([x.idx for x in node.children])

            for child in node.children:
                parents = list(self._graph.predecessors(child.idx))

                assert len(parents) == 1

                assert parents[0] == node.idx


class MarginalNode(object):
    """ A node in FS-CRP forest with parameters marginalized.
    """

    def __init__(self, idx, grid_size, children=None):
        self.idx = idx

        self.grid_size = grid_size

        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.data = []

        self.log_likelihood = np.ones(grid_size) * -np.log(grid_size[1])

        self.log_R = np.zeros(grid_size)

        self.update()

    @property
    def children(self):
        """ List of child nodes.
        """
        return self._children.values()

    @property
    def log_p(self):
        """ Log probability of sub-tree rooted at node, marginalizing node parameter.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += log_sum_exp(self.log_R[i, :])

        return log_p

    @property
    def log_p_one(self):
        """ Log probability of sub-tree rooted at node, conditioned on node parameter of one.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += self.log_R[i, -1]

        return log_p

    def add_child_node(self, node):
        """ Add a child node.
        """
        assert node.idx not in self.children

        self._children[node.idx] = node

        self.update()

    def remove_child_node(self, node):
        """ Remove a child node.
        """
        del self._children[node.idx]

        self.update()

    def update_children(self, children):
        """ Set the node children
        """
        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.log_R = np.zeros(self.grid_size)

        self.update()

    def add_data_point(self, data_point):
        """ Add a data point to the collection at this node.
        """
        self.data.append(data_point)

        self.log_likelihood += data_point.value

        self._update_log_R()

    def remove_data_point(self, data_point):
        """ Remove a data point to the collection at this node.
        """
        self.data.append(data_point)

        self.log_likelihood -= data_point.value

        self._update_log_R()

    def copy(self):
        """ Make a deep copy of the node.

        This should return a copy of the node with no shared memory with the original.
        """
        # TODO: Replace call to __init__ with __new__ and skip call to update()
        new_children = [x.copy() for x in self.children]

        new = MarginalNode(self.idx, self.grid_size, children=new_children)

        new.data = list(self.data)

        new.log_likelihood = np.copy(self.log_likelihood)

        new.log_R = np.copy(self.log_R)

        new.log_S = np.copy(self.log_S)

        return new

    def shallow_copy(self):
        new = MarginalNode.__new__(MarginalNode)

        new._children = self._children

        new.grid_size = self.grid_size

        new.idx = self.idx

        new.data = list(self.data)

        new.log_likelihood = np.copy(self.log_likelihood)

        new.log_R = np.copy(self.log_R)

        new.log_S = np.copy(self.log_S)

        return new

    def update(self):
        """ Update the arrays required for the recursion.
        """
        self._update_log_S()

        self._update_log_R()

    def _compute_log_D(self):
        for child_id, child in enumerate(self.children):
            if child_id == 0:
                log_D = child.log_R.copy()

            else:
                for i in range(self.grid_size[0]):
                    log_D[i, :] = _compute_log_D_n(child.log_R[i, :], log_D[i, :])

        return log_D

    def _update_log_R(self):
        self.log_R = self.log_likelihood + self.log_S

    def _update_log_S(self):
        self.log_S = np.zeros(self.grid_size)

        if len(self.children) > 0:
            log_D = self._compute_log_D()

            for i in range(self.grid_size[0]):
                self.log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])


def _compute_log_D_n(child_log_R, prev_log_D_n):
    """ Compute the recursion over D using the FFT
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = fftconvolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max


def get_nodes(source):
    nodes = [source, ]

    for child in source.children:
        nodes.extend(get_nodes(child))

    return nodes


def get_single_node_tree(data):
    """ Load a tree with all data points assigned single node.
    """
    nodes = [MarginalNode(0, data[0].shape, []), ]

    nodes[0].data = data

    return Tree(nodes)
