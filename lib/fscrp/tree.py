'''
Created on 16 Sep 2017

@author: Andrew Roth
'''
import networkx as nx

from fscrp.kernels.marginal.data_structures import MarginalNode


class Tree(object):

    def __init__(self, data_points, nodes):
        """ A command produced a non-zero exit code.

        :param args: dictionary mapping node ids to data points
        :param _nodes: list of _nodes

        """

        self._nodes = {}

        for n in nodes:
            self._nodes[n.idx] = n

        self._data_points = data_points

        self._init_graph()

    def _init_graph(self):
        # Create graph
        G = nx.DiGraph()

        for node in self._nodes.values():
            G.add_node(node.idx)

            for child in node.children:
                G.add_edge(node.idx, child.idx)

        # Connect roots to dummy root
        G.add_node(-1)

        for node in self._nodes.values():
            if G.in_degree(node.idx) == 0:
                G.add_edge(-1, node.idx)

        self._graph = G

        # Instantiate dummy node
        self._data_points[-1] = []

        self._nodes[-1] = MarginalNode(
            -1,
            self._nodes.values()[0].grid_size,
            children=self.roots
        )

    @property
    def data_points(self):
        data_points = self._data_points.copy()

        del data_points[-1]

        return data_points

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
    def roots(self):
        return [self._nodes[idx] for idx in self._graph.successors(-1)]

    def copy(self):
        root = self._nodes[-1].copy()

        return Tree(get_nodes(root), self._data_points.copy())

    def draw(self, ax=None):
        nx.draw(
            self.graph,
            ax=ax,
            pos=nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot'),
            with_labels=True
        )

    def get_children_nodes(self, node):
        children = []

        children_idxs = self._graph.successors(node.idx)

        for idx in children_idxs:
            children.append(self._nodes[idx])

        return children

    def get_parent_node(self, node):
        if node.idx == -1:
            return None

        parent_idxs = self._graph.predecessors(node.idx)

        assert len(parent_idxs) == 1

        parent_idx = parent_idxs[0]

        return self._nodes[parent_idx]

    def add_node(self, data_points, node, children=None, parent=None):
        if node.idx in self._nodes:
            raise Exception('Node {} exists in tree'.format(node.idx))

        self._data_points[node.idx] = data_points

        # Node editing
        self._nodes[node.idx] = node

        if children is not None:
            node.update_children(children)

            for child in children:
                self._graph.add_edge(node.idx, child.idx)

        if parent is not None:
            parent.add_child_node(node)

            self._update_ancestor_nodes(parent)

            self._graph.add_edge(parent.idx, node.idx)

    def remove_node(self, node):
        if node.idx == -1:
            raise Exception('Cannot remove root node')

        # Node editing
        parent = self.get_parent_node(node)

        parent.remove_child_node(node)

        self._update_ancestor_nodes(parent)

        del self._data_points[node.idx]

        del self._nodes[node.idx]

        # Graph editing
        for edge in self._graph.edges():
            if node.idx in edge:
                self._graph.remove_edge(*edge)

        self._graph.remove_node(node.idx)

    def add_subtree(self, subtree, parent=None):
        if parent is None:
            parent = self._nodes[-1]

        for n in subtree._nodes.values():
            if n.idx == -1:
                continue

            assert n.idx not in self._nodes

            self._data_points[n.idx] = subtree._data_points[n.idx]

            self._nodes[n.idx] = n

            for child in n.children:
                self._graph.add_edge(n.idx, child.idx)

        for node in subtree.roots:
            self._graph.add_edge(parent.idx, node.idx)

            parent.add_child_node(node)

            assert parent == self.get_parent_node(node)

            assert node in parent.children

        self._update_ancestor_nodes(parent)

    def get_subtree(self, subtree_root):
        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        print subtree_node_idxs

        data_points = {}

        nodes = []

        for node_idx in subtree_node_idxs:
            data_points[node_idx] = self._data_points[node_idx]

            nodes.append(self._nodes[node_idx])

        return Tree(data_points, nodes)

    def remove_subtree(self, subtree):
        assert len(subtree.roots) == 1

        subtree_root = subtree.roots[0]

        # Node editing
        parent = self.get_parent_node(subtree_root)

        parent.remove_child_node(subtree_root)

        assert subtree_root not in parent.children

        self._update_ancestor_nodes(parent)

        subtree_node_idxs = list(nx.dfs_tree(self._graph, subtree_root.idx))

        # Update data structures
        for node_idx in subtree_node_idxs:
            del self._data_points[node_idx]

            del self._nodes[node_idx]

        # Graph editing
        self._graph.remove_nodes_from(subtree_node_idxs)

    def _update_ancestor_nodes(self, source):
        '''
        Update all ancestor _nodes of source sequentially from source to root.
        '''
        while source is not None:
            self._nodes[source.idx].update()

            source = self.get_parent_node(source)


def get_nodes(source):
    nodes = [source, ]

    for child in source.children:
        nodes.extend(get_nodes(child))

    return nodes
