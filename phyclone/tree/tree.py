from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx

from phyclone.tree.visitors import (PostOrderNodeUpdater, PreOrderNodeRelabeller,
                                    GraphToCladesVisitor, GraphToNewickVisitor)
from phyclone.utils.math import log_factorial, cached_log_factorial
from phyclone.tree.utils import compute_log_S
import itertools
from typing import Union
from rustworkx.visualization import mpl_draw
from scipy.special import gammaln


class Tree(object):
    __slots__ = ("grid_size", "_data", "_log_prior", "_graph", "_node_indices", "_node_indices_rev",
                 "_last_node_added_to")

    def __init__(self, grid_size):
        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = rx.PyDiGraph()

        self._node_indices = dict()

        self._node_indices_rev = dict()

        self._last_node_added_to = None

        self._add_node("root")

    def __hash__(self):
        return hash((self.get_clades(), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (self.get_clades(), frozenset(self.outliers))

        other_key = (other.get_clades(), frozenset(other.outliers))

        return self_key == other_key

    def to_newick_string(self):
        vis = GraphToNewickVisitor(self)
        root_idx = self._node_indices["root"]
        rx.dfs_search(self._graph, [root_idx], vis)
        return vis.final_string

    def get_clades(self):
        vis = GraphToCladesVisitor(self)
        root_idx = self._node_indices["root"]
        rx.dfs_search(self._graph, [root_idx], vis)
        vis_clades = frozenset(vis.clades)
        return vis_clades

    def quick_draw_tree(self):
        mpl_draw(self._graph, labels=lambda node: str(node.node_id), with_labels=True)
        plt.show()
        plt.close()

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

        for data_point in data:
            tree._internal_add_data_point_to_node(True, data_point, node)

        tree.update()

        return tree

    @property
    def graph(self):
        result = self._graph.copy()

        root_idx = self._node_indices["root"]

        result.remove_node(root_idx)

        return result

    @property
    def data(self):
        result = sorted(itertools.chain.from_iterable(self._data.values()), key=lambda x: x.idx)
        return result

    @property
    def data_log_likelihood(self):
        """The log likelihood grid of the data for all values of the root node."""
        root_idx = self._node_indices["root"]
        return self._graph[root_idx].log_r

    @property
    def labels(self):
        result = {dp.idx: k for k, l in self.node_data.items() for dp in l}
        return result

    @property
    def leafs(self):
        return [x for x in self.nodes if self.get_number_of_children(x) == 0]

    # @property
    # def multiplicity(self):
    #     return self._get_multiplicity("root")
    #
    # def _get_multiplicity(self, node):
    #     children = self.get_children(node)
    #
    #     result = log_factorial(len(children))
    #
    #     for child in children:
    #         result += self._get_multiplicity(child)
    #
    #     return result

    @property
    def multiplicity(self):
        mult = sum(map(cached_log_factorial, map(self._graph.out_degree, self._graph.node_indices())))
        return mult

    @property
    def nodes(self):
        result = [node.node_id for node in self._graph.nodes() if node.node_id != "root"]
        return result

    @property
    def node_last_added_to(self):
        return self._last_node_added_to

    def get_number_of_nodes(self):
        return self._graph.num_nodes() - 1

    @property
    def node_data(self):
        result = self._data.copy()

        if "root" in result:
            del result["root"]

        return result

    @property
    def outliers(self):
        return list(self._data[-1])

    @property
    def roots(self):
        node_idx = self._node_indices["root"]
        return [child.node_id for child in self._graph.successors(node_idx)]

    @staticmethod
    def from_dict(tree_dict):
        grid_size = tree_dict["grid_size"]
        new = Tree(grid_size)

        if len(tree_dict["graph"]) > 0:

            new_graph = new._graph
            node_idxs = tree_dict['node_idx']
            # assert node_idxs["root"] == new._node_indices["root"]

            new_graph.extend_from_edge_list(tree_dict['graph'])

            log_prior = new._log_prior
            new_data = new._data

            for node, data_list in tree_dict["node_data"].items():
                new_data[node] = list(data_list)
                if node == -1 or node == "root":
                    continue

                log_p = np.full(grid_size, log_prior, order="C")

                for dp in data_list:
                    log_p += dp.value

                node_obj = TreeNode(log_p,
                                    np.zeros(grid_size, order="C"),
                                    node)
                node_idx = node_idxs[node]
                new_graph[node_idx] = node_obj

            new._node_indices_rev = tree_dict['node_idx_rev'].copy()
            new._node_indices = node_idxs.copy()

            # node_index_holes = set(new_graph.node_indices()).difference(tree_dict['node_idx_rev'].keys())

            node_index_holes = [idx for idx in new_graph.node_indices() if idx not in tree_dict['node_idx_rev']]

            if len(node_index_holes) > 0:
                new_graph.remove_nodes_from(node_index_holes)

        else:
            for node, data_list in tree_dict["node_data"].items():
                new._data[node] = list(data_list)

        new._last_node_added_to = tree_dict['node_last_added_to']

        new.update()

        return new

    def to_dict(self):
        node_data = {k: v.copy() for k, v in self._data.items() if k != "root"}

        res = {"graph": self._graph.edge_list(),
               "node_idx": self._node_indices.copy(),
               "node_idx_rev": self._node_indices_rev.copy(),
               "node_data": node_data,
               "grid_size": self.grid_size,
               "node_last_added_to": self._last_node_added_to}
        return res

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._internal_add_data_point_to_node(False, data_point, node)

    def _internal_add_data_point_to_node(self, build_add, data_point, node):
        self._data[node].append(data_point)
        if node != -1:
            node_idx = self._node_indices[node]

            self._graph[node_idx].log_p += data_point.value

            self._graph[node_idx].log_r += data_point.value

            self._last_node_added_to = node

            if not build_add:
                self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)
        self._last_node_added_to = -1

    # def add_subtree(self, subtree, parent=None):
    #
    #     subtree = subtree.copy()
    #
    #     # Connect subtree
    #     if parent is None:
    #         parent = "root"
    #
    #     parent_idx = self._node_indices[parent]
    #
    #     parent_node = self._graph[parent_idx]
    #
    #     subtree_root = subtree.roots[0]
    #
    #     subtree_dummy_root = subtree._node_indices["root"]
    #
    #     subtree._graph.remove_node(subtree_dummy_root)
    #
    #     node_map_idx = self._graph.compose(subtree._graph, {parent_idx: (subtree._node_indices[subtree_root], None)})
    #
    #     for old_idx, new_idx in node_map_idx.items():
    #         node_obj = self._graph[new_idx]
    #         node_name = node_obj.node_id
    #         # assert node_name not in self._data
    #         self._data[node_name] = subtree._data[node_name]
    #         self._node_indices[node_name] = new_idx
    #         self._node_indices_rev[new_idx] = node_name
    #
    #     self._update_path_to_root(parent_node.node_id)

    def add_subtree(self, subtree, parent=None):

        subtree = subtree.copy()

        first_label = (max(self.nodes + subtree.nodes + [-1, ]) + 1) - 1

        # Connect subtree
        if parent is None:
            parent = "root"

        parent_idx = self._node_indices[parent]

        parent_node = self._graph[parent_idx]

        subtree_dummy_root = subtree._node_indices["root"]

        node_map_idx = self._graph.compose(subtree._graph, {parent_idx: (subtree_dummy_root, None)})

        self._graph.remove_node_retain_edges(node_map_idx[subtree_dummy_root])

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
            self._node_indices[node_name] = new_idx
            self._node_indices_rev[new_idx] = node_name

        self._last_node_added_to = subtree._last_node_added_to

        self._update_path_to_root(parent_node.node_id)

    def create_root_node(self, children=[], data=[]):
        """Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        data: list
            Data points to add to new node.
        """
        node = self._graph.num_nodes() - 1

        self._add_node(node)

        node_idx = self._node_indices[node]

        root_idx = self._node_indices["root"]

        for data_point in data:
            self._data[node].append(data_point)

            self._graph[node_idx].log_p += data_point.value

        self._graph.add_edge(root_idx, node_idx, None)

        for child in children:
            child_idx = self._node_indices[child]
            self._graph.remove_edge(root_idx, child_idx)

            self._graph.add_edge(node_idx, child_idx, None)

        self._last_node_added_to = node

        self._update_path_to_root(node)

        return node

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

    def get_descendants(self, source="root"):
        source_idx = self._node_indices[source]
        descs = rx.descendants(self._graph, source_idx)
        return [self._graph[child].node_id for child in descs]

    def get_number_of_descendants(self, source="root"):
        source_idx = self._node_indices[source]
        descs = rx.descendants(self._graph, source_idx)
        return len(list(descs))

    def get_parent(self, node):
        if node == "root":
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
        if subtree_root == "root":
            return self.copy()

        new = Tree(self.grid_size)

        subtree_root_idx = self._node_indices[subtree_root]

        subtree_graph_node_indices = [subtree_root_idx] + list(rx.descendants(self._graph, subtree_root_idx))

        subtree_graph = self._graph.subgraph(subtree_graph_node_indices, preserve_attrs=True)

        new_root_idx = new._node_indices["root"]

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

            new._node_indices[node] = node_idx
            new._node_indices_rev[node_idx] = node

        new.update()

        return new

    def get_subtree_data(self, node):
        data = self.get_data(node)

        for desc in self.get_descendants(node):
            data.extend(self.get_data(desc))

        return data

    def relabel_nodes(self):
        data = defaultdict(list)

        data[-1] = self._data[-1]

        vis = PreOrderNodeRelabeller(self, data)

        root_idx = self._node_indices["root"]

        rx.dfs_search(self._graph, [root_idx], vis)

        self._data = data

        self._node_indices = vis.node_indices
        self._node_indices_rev = vis.node_indices_rev

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        if node != -1:
            node_idx = self._node_indices[node]
            self._graph[node_idx].log_p -= data_point.value

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
                if node_id != "root":
                    del self._data[node_id]
                    curr_idx = self._node_indices[node_id]
                    del self._node_indices[node_id]
                    del self._node_indices_rev[curr_idx]

            indices_to_remove = list(rx.descendants(self._graph, sub_root_idx)) + [sub_root_idx]

            self._graph.remove_nodes_from(indices_to_remove)

            self._update_path_to_root(parent_node.node_id)

    def update(self):
        vis = PostOrderNodeUpdater(self)
        root_idx = self._node_indices["root"]
        rx.dfs_search(self._graph, [root_idx], vis)

    def _add_node(self, node):
        node_obj = TreeNode(np.full(self.grid_size, self._log_prior, order="C"),
                            np.zeros(self.grid_size, order="C"),
                            node)
        new_node = self._graph.add_node(node_obj)
        self._node_indices[node] = new_node
        self._node_indices_rev[new_node] = node

    def _update_path_to_root(self, source):
        """Update recursion values for all nodes on the path between the source node and root inclusive."""
        root_idx = self._node_indices["root"]
        source_idx = self._node_indices[source]
        paths = rx.all_simple_paths(self._graph, root_idx, source_idx)

        if len(paths) == 0:
            assert source == "root"
            paths = [[root_idx]]

        assert len(paths) == 1

        path = paths[0]

        assert self._node_indices_rev[path[-1]] == source

        assert self._node_indices_rev[path[0]] == "root"

        for source in reversed(path):
            self._update_node(source)

    def _update_node(self, node_idx):
        child_log_r_values = [child.log_r for child in self._graph.successors(node_idx)]

        log_p = self._graph[node_idx].log_p

        if len(child_log_r_values) == 0:
            np.copyto(self._graph[node_idx].log_r, log_p)
            return
        else:
            log_s = compute_log_S(child_log_r_values)

        self._graph[node_idx].log_r = np.add(log_p, log_s, out=self._graph[node_idx].log_r, order="C")


class TreeNode(object):
    __slots__ = ("log_p", "log_r", "node_id")

    def __init__(self, log_p: np.array, log_r: np.array, node_id: Union[str | int]):
        self.log_p = log_p
        self.log_r = log_r
        self.node_id = node_id

    def __copy__(self):
        return TreeNode(self.log_p.copy(),
                        self.log_r.copy(),
                        self.node_id)

    def __eq__(self, other):
        log_p_compare = np.array_equal(self.log_p, other.log_p)
        log_r_compare = np.array_equal(self.log_r, other.log_r)
        # id_compare = self.node_id == other.node_id
        return log_p_compare and log_r_compare

    def copy(self):
        return TreeNode(self.log_p.copy(),
                        self.log_r.copy(),
                        self.node_id)

    def to_dict(self):
        return {"log_p": self.log_p, "log_R": self.log_r, "node_id": self.node_id}
