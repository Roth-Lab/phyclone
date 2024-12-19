from collections import defaultdict
from itertools import chain

import numpy as np
import rustworkx as rx

from phyclone.tree.tree_node import TreeNode
from phyclone.tree.visitors import (
    PostOrderNodeUpdater,
    PreOrderNodeRelabeller,
    GraphToCladesVisitor,
    GraphToNewickVisitor,
)
from phyclone.utils.math import cached_log_factorial


class Tree(object):
    _ROOT_NODE_NAME = "root"
    __slots__ = (
        "grid_size",
        "_data",
        "_log_prior",
        "_graph",
        "_node_indices",
        "_node_indices_rev",
        "_last_node_added_to",
    )

    def __init__(self, grid_size):
        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = rx.PyDiGraph()

        self._node_indices = dict()

        self._node_indices_rev = dict()

        self._last_node_added_to = None

        self._add_node(self._ROOT_NODE_NAME)

    def __hash__(self):
        return hash((self.get_clades(), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (self.get_clades(), frozenset(self.outliers))

        other_key = (other.get_clades(), frozenset(other.outliers))

        return self_key == other_key

    def to_newick_string(self):
        visitor = GraphToNewickVisitor(self)
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        rx.dfs_search(self._graph, [root_idx], visitor)
        return visitor.final_string

    def get_clades(self):
        visitor = GraphToCladesVisitor(self)
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        rx.dfs_search(self._graph, [root_idx], visitor)
        vis_clades = frozenset(visitor.clades)
        return vis_clades

    @staticmethod
    def get_single_node_tree(data):
        """Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        tree = Tree(data[0].grid_size)

        node = tree.create_root_node([])

        _ = tree._add_list_of_data_points_to_node(data, node)

        tree.update()

        return tree
    
    @property
    def root_node_name(self):
        return self._ROOT_NODE_NAME

    @property
    def graph(self):
        result = self._graph.copy()

        root_idx = self._node_indices[self._ROOT_NODE_NAME]

        result.remove_node(root_idx)

        return result

    @property
    def data(self):
        result = sorted(chain.from_iterable(self._data.values()), key=lambda x: x.idx)
        return result

    @property
    def data_log_likelihood(self):
        """The log likelihood grid of the data for all values of the root node."""
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        return self._graph[root_idx].log_r

    @property
    def labels(self):
        result = {dp.idx: k for k, l in self.node_data.items() for dp in l}
        return result

    @property
    def multiplicity(self):
        mult = sum(
            map(
                cached_log_factorial,
                map(self._graph.out_degree, self._graph.node_indices()),
            )
        )
        return mult

    @property
    def nodes(self):
        result = [node.node_id for node in self._graph.nodes() if node.node_id != self._ROOT_NODE_NAME]
        return result

    @property
    def node_last_added_to(self):
        return self._last_node_added_to

    def get_number_of_nodes(self):
        return self._graph.num_nodes() - 1

    @property
    def node_data(self):
        result = self._data.copy()

        if self._ROOT_NODE_NAME in result:
            del result[self._ROOT_NODE_NAME]

        return result

    @property
    def outliers(self):
        return list(self._data[-1])

    @property
    def roots(self):
        node_idx = self._node_indices[self._ROOT_NODE_NAME]
        return [child.node_id for child in self._graph.successors(node_idx)]

    @classmethod
    def from_dict(cls, tree_dict):
        grid_size = tree_dict["grid_size"]
        if "log_prior" in tree_dict:
            log_prior = tree_dict["log_prior"]
        else:
            log_prior = -np.log(grid_size[1])
        new_graph = rx.PyDiGraph()

        new = cls.__new__(cls)
        new.grid_size = grid_size
        new._graph = new_graph
        new._log_prior = log_prior
        new._data = defaultdict(list)

        new._node_indices_rev = tree_dict["node_idx_rev"].copy()
        new._node_indices = tree_dict["node_idx"].copy()
        new._last_node_added_to = tree_dict["node_last_added_to"]
        new._data.update({k: v.copy() for k, v in tree_dict["node_data"].items()})

        _ = new_graph.add_node(TreeNode(grid_size, log_prior, cls._ROOT_NODE_NAME))

        if len(tree_dict["graph"]) > 0:

            node_idxs = tree_dict["node_idx"]
            new_graph.extend_from_edge_list(tree_dict["graph"])
            root_name = cls._ROOT_NODE_NAME

            for node, data_list in tree_dict["node_data"].items():
                if node == -1 or node == root_name:
                    continue
                node_obj = TreeNode(grid_size, log_prior, node)
                node_obj.add_data_point_list(data_list)
                node_idx = node_idxs[node]
                new_graph[node_idx] = node_obj

            node_index_holes = [idx for idx in new_graph.node_indices() if idx not in tree_dict["node_idx_rev"]]
            if len(node_index_holes) > 0:
                new_graph.remove_nodes_from(node_index_holes)

        new.update()
        return new

    def to_dict(self):
        tree_dict = {
            "graph": self._graph.edge_list(),
            "node_idx": self._node_indices.copy(),
            "node_idx_rev": self._node_indices_rev.copy(),
            "node_data": {k: v.copy() for k, v in self._data.items()},
            "grid_size": self.grid_size,
            "node_last_added_to": self._last_node_added_to,
            "log_prior": self._log_prior,
        }
        return tree_dict

    def serialize_tree(self):
        serial = rx.node_link_json(self._graph, node_attrs=lambda x: x.serialize())
        return serial

    def _is_data_point_in_tree(self, data_point):
        dp_idx = data_point.idx
        data_point_is_present = sum(map(lambda x: dp_idx in x.data_points, self._graph.nodes()))
        if -1 in self._data:
            dp_in_outliers = data_point in self._data[-1]
            data_point_is_present += dp_in_outliers
        return data_point_is_present

    def add_data_point_to_node(self, data_point, node):
        assert self._is_data_point_in_tree(data_point) == False

        self._internal_add_data_point_to_node(False, data_point, node)

    def _internal_add_data_point_to_node(self, build_add, data_point, node):
        self._data[node].append(data_point)
        self._last_node_added_to = node

        if node != -1:
            node_idx = self._node_indices[node]

            self._graph[node_idx].add_data_point(data_point)

            if not build_add:
                self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self.add_data_point_to_node(data_point, -1)

    def add_subtree(self, subtree, parent=None):

        subtree = subtree.copy()

        # Connect subtree
        if parent is None:
            parent = self._ROOT_NODE_NAME
        parent_idx = self._node_indices[parent]
        parent_node = self._graph[parent_idx]
        subtree_dummy_root = subtree._node_indices[subtree._ROOT_NODE_NAME]
        node_map_idx = self._graph.compose(subtree._graph, {parent_idx: (subtree_dummy_root, None)})

        self._graph.remove_node_retain_edges(node_map_idx[subtree_dummy_root])

        self._relabel_grafted_subtree_nodes(node_map_idx, subtree, subtree_dummy_root)

        self._last_node_added_to = subtree._last_node_added_to

        self._update_path_to_root(parent_node.node_id)

    def _relabel_grafted_subtree_nodes(self, node_map_idx, subtree, subtree_dummy_root):
        first_label = (max(self.nodes + subtree.nodes + [-1]) + 1) - 1

        for old_idx, new_idx in node_map_idx.items():
            if old_idx == subtree_dummy_root:
                continue
            node_obj = self._graph[new_idx]
            node_name = node_obj.node_id
            old_node_name = node_name
            if node_name in self._data:
                first_label += 1
                node_name = first_label
                node_obj.node_id = node_name
            self._data[node_name] = subtree._data[old_node_name]
            self._add_node_to_indices(node_name, new_idx)

    def create_root_node(self, children=None, data=None):
        """Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        data: list
            Data points to add to new node.
        """
        if data is None:
            data = []
        if children is None:
            children = []

        node = self._graph.num_nodes() - 1

        self._add_node(node)

        root_idx = self._node_indices[self._ROOT_NODE_NAME]

        node_idx = self._add_list_of_data_points_to_node(data, node)

        self._graph.add_edge(root_idx, node_idx, None)

        for child in children:
            child_idx = self._node_indices[child]
            self._graph.remove_edge(root_idx, child_idx)

            self._graph.add_edge(node_idx, child_idx, None)

        self._last_node_added_to = node

        self._update_path_to_root(node)

        return node

    def _add_list_of_data_points_to_node(self, data, node):
        node_idx = self._node_indices[node]
        if len(data) > 0:
            self._data[node].extend(data)
            self._graph[node_idx].add_data_point_list(data)
        return node_idx

    def copy(self):
        cls = self.__class__

        new = cls.__new__(cls)

        new.grid_size = self.grid_size

        new._data = defaultdict(list)

        new._data.update({k: v.copy() for k, v in self._data.items()})

        new._log_prior = self._log_prior

        new._graph = self._graph.copy()

        new._node_indices = self._node_indices.copy()

        new._node_indices_rev = self._node_indices_rev.copy()

        new._last_node_added_to = self._last_node_added_to

        for node_idx in new._graph.node_indices():
            new._graph[node_idx] = new._graph[node_idx].copy()

        return new

    def get_children(self, node):
        node_idx = self._node_indices[node]
        return [child.node_id for child in self._graph.successors(node_idx)]

    def get_number_of_children(self, node):
        node_idx = self._node_indices[node]
        return len(self._graph.successors(node_idx))

    def get_descendants(self, source=None):
        if source is None:
            source = self._ROOT_NODE_NAME
        source_idx = self._node_indices[source]
        descs = rx.descendants(self._graph, source_idx)
        return [self._graph[child].node_id for child in descs]

    def get_number_of_descendants(self, source=None):
        if source is None:
            source = self._ROOT_NODE_NAME
        source_idx = self._node_indices[source]
        descs = rx.descendants(self._graph, source_idx)
        return len(descs)

    def get_parent(self, node):
        if node == self._ROOT_NODE_NAME:
            return None
        else:
            node_idx = self._node_indices[node]
            return [pred.node_id for pred in self._graph.predecessors(node_idx)][0]

    def get_data(self, node):
        return list(self._data[node])

    def get_data_len(self, node):
        return len(self._data[node])

    def get_subtree_data_len(self, node):
        data_len = self.get_data_len(node)

        for desc in self.get_descendants(node):
            data_len += self.get_data_len(desc)

        return data_len

    def get_subtree(self, subtree_root):
        if subtree_root == self._ROOT_NODE_NAME:
            return self.copy()

        new = Tree(self.grid_size)

        subtree_root_idx = self._node_indices[subtree_root]

        subtree_graph_node_indices = [subtree_root_idx] + list(rx.descendants(self._graph, subtree_root_idx))

        subtree_graph = self._graph.subgraph(subtree_graph_node_indices, preserve_attrs=True)

        new_root_idx = new._node_indices[self._ROOT_NODE_NAME]

        sub_root_idx = -1

        for sub_idx in subtree_graph.node_indices():
            payload = subtree_graph[sub_idx]
            if payload.node_id == subtree_root:
                sub_root_idx = sub_idx
                break

        new._graph.compose(subtree_graph, {new_root_idx: (sub_root_idx, None)})

        for node_idx in new._graph.node_indices():
            new._graph[node_idx] = new._graph[node_idx].copy()
            node = new._graph[node_idx].node_id
            new._data[node] = list(self._data[node])

            new._add_node_to_indices(node, node_idx)

        new.update()

        return new

    def relabel_nodes(self):
        data = defaultdict(list)
        data[-1] = list(self._data[-1])

        visitor = PreOrderNodeRelabeller(self, data)
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        rx.dfs_search(self._graph, [root_idx], visitor)

        self._data = data
        self._node_indices = visitor.node_indices
        self._node_indices_rev = visitor.node_indices_rev

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        if node != -1:
            node_idx = self._node_indices[node]
            self._graph[node_idx].remove_data_point(data_point)

            self._update_path_to_root(node)

    def remove_data_point_from_outliers(self, data_point):
        self._data[-1].remove(data_point)

    def remove_subtree(self, subtree):
        if subtree == self:
            self.__init__(self.grid_size)

        else:
            assert len(subtree.roots) == 1

            sub_root = subtree.roots[0]

            parent = self.get_parent(sub_root)
            parent_idx = self._node_indices[parent]
            parent_node = self._graph[parent_idx]

            sub_root_idx = self._node_indices[sub_root]

            for node in subtree._graph.nodes():
                node_id = node.node_id
                if node_id != self._ROOT_NODE_NAME:
                    del self._data[node_id]
                    curr_idx = self._node_indices[node_id]
                    del self._node_indices[node_id]
                    del self._node_indices_rev[curr_idx]

            indices_to_remove = list(rx.descendants(self._graph, sub_root_idx)) + [sub_root_idx]
            self._graph.remove_nodes_from(indices_to_remove)
            self._update_path_to_root(parent_node.node_id)

    def update(self):
        """Update recursion values for all nodes in the tree."""
        vis = PostOrderNodeUpdater(self._update_node)
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        rx.dfs_search(self._graph, [root_idx], vis)

    def _add_node(self, node):
        node_obj = TreeNode(self.grid_size, self._log_prior, node)
        node_idx = self._graph.add_node(node_obj)
        self._add_node_to_indices(node, node_idx)

    def _add_node_to_indices(self, node, node_idx):
        self._node_indices[node] = node_idx
        self._node_indices_rev[node_idx] = node

    def _update_path_to_root(self, source):
        """Update recursion values for all nodes on the path between the source node and root inclusive."""
        root_idx = self._node_indices[self._ROOT_NODE_NAME]
        source_idx = self._node_indices[source]
        paths = rx.all_simple_paths(self._graph, root_idx, source_idx)

        if len(paths) == 0:
            assert source == self._ROOT_NODE_NAME
            paths = [[root_idx]]

        assert len(paths) == 1
        path = paths[0]

        assert self._node_indices_rev[path[-1]] == source
        assert self._node_indices_rev[path[0]] == self._ROOT_NODE_NAME

        for source in reversed(path):
            self._update_node(source)

    def _update_node(self, node_idx):
        child_log_r_values = [child.log_r for child in self._graph.successors(node_idx)]

        self._graph[node_idx].update_node_from_child_r_vals(child_log_r_values)
