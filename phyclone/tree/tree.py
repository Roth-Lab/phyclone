from collections import defaultdict
from copy import copy
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import rustworkx as rx

from phyclone.utils.math import log_factorial
from phyclone.tree.utils import compute_log_S, get_clades
from phyclone.utils import get_iterator_length
import itertools
from rustworkx.visit import DFSVisitor
from typing import Union
from rustworkx.visualization import mpl_draw


class Tree(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        # self._graph = nx.DiGraph()
        self._graph = rx.PyDiGraph()

        self._node_indices = dict()

        self._node_indices_rev = dict()

        self._add_node("root")

    def __hash__(self):
        return hash((get_clades(self), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (get_clades(self), frozenset(self.outliers))

        other_key = (get_clades(other), frozenset(other.outliers))

        return self_key == other_key

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
        # return self._graph.nodes["root"]["log_r"]
        root_idx = self._node_indices["root"]
        return self._graph[root_idx].log_r

    @property
    def labels(self):
        result = {dp.idx: k for k, l in self.node_data.items() for dp in l}
        return result

    @property
    def leafs(self):
        return [x for x in self.nodes if self.get_number_of_children(x) == 0]

    @property
    def multiplicity(self):
        return self._get_multiplicity("root")

    def _get_multiplicity(self, node):
        children = self.get_children(node)

        result = log_factorial(len(children))

        for child in children:
            result += self._get_multiplicity(child)

        return result

    @property
    def nodes(self):
        # result = list(self._graph.nodes())
        #
        # result.remove("root")

        result = [node.node_id for node in self._graph.nodes() if node.node_id != "root"]

        return result

    def get_number_of_nodes(self):
        # return self._graph.number_of_nodes() - 1
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
        return self.get_children("root")
        # return list(self._graph.successors("root"))

    # @staticmethod
    # def from_dict(data, tree_dict):
    #     new = Tree(data[0].grid_size)
    #
    #     # new._graph = nx.DiGraph(tree_dict["graph"])
    #     new._graph = nx.DiGraph(tree_dict["graph"])
    #
    #     data = dict(zip([x.idx for x in data], data))
    #
    #     for node in new._graph.nodes:
    #         new._add_node(node)
    #
    #     for idx, node in tree_dict["labels"].items():
    #         new._internal_add_data_point_to_node(True, data[idx], node)
    #
    #     new.update()
    #
    #     return new
    @staticmethod
    def from_dict(data, tree_dict):
        new = Tree(data[0].grid_size)

        data = dict(zip([x.idx for x in data], data))

        for node in tree_dict["graph"].keys():
            if node == "root":
                continue
            new._add_node(node)

        for parent, children in tree_dict["graph"].items():
            parent_idx = new._node_indices[parent]
            for child in children.keys():
                child_idx = new._node_indices[child]
                new._graph.add_edge(parent_idx, child_idx, None)

        for idx, node in tree_dict["labels"].items():
            new._internal_add_data_point_to_node(True, data[idx], node)

        new.update()

        return new

    def to_dict(self):
        # GraphToDictVisitor
        vis = GraphToDictVisitor(self)
        root_idx = self._node_indices["root"]
        rx.dfs_search(self._graph, [root_idx], vis)
        res = {"graph": vis.dict_of_dicts, "labels": self.labels}
        return res
        # return {"graph": nx.to_dict_of_dicts(self._graph), "labels": self.labels}

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._internal_add_data_point_to_node(False, data_point, node)

    def _internal_add_data_point_to_node(self, build_add, data_point, node):
        self._data[node].append(data_point)
        if node != -1:
            # self._graph.nodes[node]["log_p"] += data_point.value
            #
            # self._graph.nodes[node]["log_r"] += data_point.value

            node_idx = self._node_indices[node]

            self._graph[node_idx].log_p += data_point.value

            self._graph[node_idx].log_r += data_point.value

            if not build_add:
                self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)

    def add_subtree(self, subtree, parent=None):
        first_label = (max(self.nodes + subtree.nodes + [-1, ]) + 1)

        node_map = {}
        #
        subtree = subtree.copy()
        #
        for new_node, old_node in enumerate(subtree.nodes, first_label):
            node_map[old_node] = new_node

            self._data[new_node] = subtree._data[old_node]

        # Connect subtree
        if parent is None:
            parent = "root"

        parent_idx = self._node_indices[parent]

        subtree_root = subtree.roots[0]

        subtree_dummy_root = subtree._node_indices["root"]

        subtree._graph.remove_node(subtree_dummy_root)

        node_map_idx = self._graph.compose(subtree._graph, {parent_idx: (subtree._node_indices[subtree_root], None)})

        for subtree_idx, tree_idx in node_map_idx.items():
            old_node_id = self._graph[tree_idx].node_id
            self._graph[tree_idx].node_id = node_map[old_node_id]
            self._node_indices[node_map[old_node_id]] = tree_idx
            self._node_indices_rev[tree_idx] = node_map[old_node_id]

        # nx.relabel_nodes(subtree._graph, node_map, copy=False)

        # self._graph = nx.compose(self._graph, subtree.graph)
        #
        # # Connect subtree
        # if parent is None:
        #     parent = "root"
        #
        # for node in subtree.roots:
        #     self._graph.add_edge(parent, node)

        self._update_path_to_root(parent)

    # def add_subtree(self, subtree, parent=None):
    #     first_label = (max(self.nodes + subtree.nodes + [-1, ]) + 1)
    #
    #     node_map = {}
    #
    #     subtree = subtree.copy()
    #
    #     for new_node, old_node in enumerate(subtree.nodes, first_label):
    #         node_map[old_node] = new_node
    #
    #         self._data[new_node] = subtree._data[old_node]
    #
    #     nx.relabel_nodes(subtree._graph, node_map, copy=False)
    #
    #     self._graph = nx.compose(self._graph, subtree.graph)
    #
    #     # Connect subtree
    #     if parent is None:
    #         parent = "root"
    #
    #     for node in subtree.roots:
    #         self._graph.add_edge(parent, node)
    #
    #     self._update_path_to_root(parent)

    def create_root_node(self, children=[], data=[]):
        """Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        data: list
            Data points to add to new node.
        """
        # node = nx.number_of_nodes(self._graph) - 1
        node = self._graph.num_nodes() - 1

        self._add_node(node)

        node_idx = self._node_indices[node]

        root_idx = self._node_indices["root"]

        for data_point in data:
            self._data[node].append(data_point)

            # self._graph.nodes[node]["log_p"] += data_point.value
            self._graph[node_idx].log_p += data_point.value

        # self._graph.add_edge("root", node)
        self._graph.add_edge(root_idx, node_idx, None)

        for child in children:
            child_idx = self._node_indices[child]
            self._graph.remove_edge(root_idx, child_idx)

            self._graph.add_edge(node_idx, child_idx, None)

        self._update_path_to_root(node)

        return node

    def copy(self):
        cls = self.__class__

        new = cls.__new__(cls)

        new.grid_size = self.grid_size

        new._data = defaultdict(list)

        for node in self._data:
            new._data[node] = list(self._data[node])

        new._log_prior = self._log_prior

        new._graph = self._graph.copy()

        new._node_indices = self._node_indices.copy()

        new._node_indices_rev = self._node_indices_rev.copy()

        for node_idx in new._graph.node_indices():
            new._graph[node_idx] = copy(new._graph[node_idx])

            # for node in new._graph:
        #     new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()
        #
        #     new._graph.nodes[node]["log_r"] = self._graph.nodes[node]["log_r"].copy()

        return new

    def get_children(self, node):
        # return list(self._graph.successors(node))
        node_idx = self._node_indices[node]
        return [child.node_id for child in self._graph.successors(node_idx)]

    def get_number_of_children(self, node):
        # children = self._graph.successors(node)
        # num_children = get_iterator_length(children)
        node_idx = self._node_indices[node]
        return len(self._graph.successors(node_idx))

    def get_descendants(self, source="root"):
        # return nx.descendants(self._graph, source=source)
        source_idx = self._node_indices[source]
        descs = rx.descendants(self._graph, source_idx)
        return [self._graph[child].node_id for child in descs]

    def get_parent(self, node):
        if node == "root":
            return None

        else:
            node_idx = self._node_indices[node]
            # return list(self._graph.predecessors(node_idx))[0]
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
            new._graph[node_idx] = copy(new._graph[node_idx])
            node = new._graph[node_idx].node_id
            new._data[node] = list(self._data[node])

            new._node_indices[node] = node_idx
            new._node_indices_rev[node_idx] = node

        new.update()

        return new

    # def get_subtree(self, subtree_root):
    #     if subtree_root == "root":
    #         return self.copy()
    #
    #     new = Tree(self.grid_size)
    #
    #     subtree_graph = nx.dfs_tree(self._graph, subtree_root)
    #
    #     new._graph = nx.compose(new._graph, subtree_graph)
    #
    #     new._graph.add_edge("root", subtree_root)
    #
    #     for node in new.nodes:
    #         new._data[node] = list(self._data[node])
    #
    #         new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()
    #
    #     new.update()
    #
    #     return new

    def get_subtree_data(self, node):
        data = self.get_data(node)

        for desc in self.get_descendants(node):
            data.extend(self.get_data(desc))

        return data

    # def relabel_nodes(self):
    #     node_map = {}
    #
    #     data = defaultdict(list)
    #
    #     data[-1] = self._data[-1]
    #
    #     new_node = 0
    #
    #     for old_node in nx.dfs_preorder_nodes(self._graph, source="root"):
    #         if old_node == "root":
    #             continue
    #
    #         node_map[old_node] = new_node
    #
    #         data[new_node] = self._data[old_node]
    #
    #         new_node += 1
    #
    #     self._data = data
    #
    #     self._graph = nx.relabel_nodes(self._graph, node_map)

    def relabel_nodes(self):
        node_map = {}

        data = defaultdict(list)

        data[-1] = self._data[-1]

        vis = PreOrderNodeRelabeller(self, data)

        root_idx = self._node_indices["root"]

        rx.dfs_search(self._graph, [root_idx], vis)

        # new_node = 0
        #
        # for old_node in nx.dfs_preorder_nodes(self._graph, source="root"):
        #     if old_node == "root":
        #         continue
        #
        #     node_map[old_node] = new_node
        #
        #     data[new_node] = self._data[old_node]
        #
        #     new_node += 1

        self._data = data

        self._node_indices = vis.node_indices
        self._node_indices_rev = vis.node_indices_rev

        # self._graph = nx.relabel_nodes(self._graph, node_map)

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        if node != -1:
            # self._graph.nodes[node]["log_p"] -= data_point.value
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

            # indices_to_remove = []
            #
            # for node in subtree._graph.nodes():
            #     indices_to_remove.append(self._node_indices[node.node_id])
            #     del self._data[node.node_id]

            for node in subtree._graph.nodes():
                node_id = node.node_id
                if node_id == "root":
                    continue
                del self._data[node_id]
                curr_idx = self._node_indices[node_id]
                del self._node_indices[node_id]
                del self._node_indices_rev[curr_idx]

            indices_to_remove = list(rx.descendants(self._graph, parent_idx))

            self._graph.remove_nodes_from(indices_to_remove)

            # parent = self.get_parent(subtree.roots[0])
            #
            # self._graph.remove_nodes_from(subtree.nodes)
            #
            # for node in subtree.nodes:
            #     del self._data[node]
            #
            self._update_path_to_root(parent)

    # def remove_subtree(self, subtree):
    #     if subtree == self:
    #         self.__init__(self.grid_size)
    #
    #     else:
    #         assert len(subtree.roots) == 1
    #
    #         parent = self.get_parent(subtree.roots[0])
    #
    #         self._graph.remove_nodes_from(subtree.nodes)
    #
    #         for node in subtree.nodes:
    #             del self._data[node]
    #
    #         self._update_path_to_root(parent)

    def update(self):
        vis = PostOrderNodeUpdater(self)
        root_idx = self._node_indices["root"]
        rx.dfs_search(self._graph, [root_idx], vis)
        # for node in nx.dfs_postorder_nodes(self._graph, "root"):
        #     self._update_node(node)

    def _add_node(self, node):
        # new_node = self._graph.add_node(node)
        node_obj = TreeNode(np.full(self.grid_size, self._log_prior, order="C"),
                            np.zeros(self.grid_size, order="C"),
                            node)
        new_node = self._graph.add_node(node_obj)
        self._node_indices[node] = new_node
        self._node_indices_rev[new_node] = node
        # self._graph.nodes[node]["log_p"] = np.full(self.grid_size, self._log_prior, order="C")
        # self._graph.nodes[node]["log_r"] = np.zeros(self.grid_size, order="C")

    def _update_path_to_root(self, source):
        """Update recursion values for all nodes on the path between the source node and root inclusive."""
        # paths = list(nx.all_simple_paths(self._graph, "root", source))
        root_idx = self._node_indices["root"]
        source_idx = self._node_indices[source]
        paths = rx.all_simple_paths(self._graph, root_idx, source_idx)

        if len(paths) == 0:
            assert source == "root"

            # paths = [["root"]]
            paths = [[root_idx]]

        assert len(paths) == 1

        path = paths[0]

        # assert path[-1] == source
        #
        # assert path[0] == "root"

        assert self._node_indices_rev[path[-1]] == source

        assert self._node_indices_rev[path[0]] == "root"

        for source in reversed(path):
            source_idx = self._node_indices_rev[source]
            self._update_node(source_idx)

    def _update_node(self, node):
        node_idx = self._node_indices[node]
        child_log_r_values = [
            child.log_r for child in self._graph.successors(node_idx)
        ]

        log_p = self._graph[node_idx].log_p

        if len(child_log_r_values) == 0:
            self._graph[node_idx].log_r = np.copy(log_p)
            return
        else:
            log_s = compute_log_S(child_log_r_values)

        # if "log_r" in self._graph.nodes[node]:
        #     self._graph.nodes[node]["log_r"] = np.add(log_p, log_s, out=self._graph.nodes[node]["log_r"], order="C")
        # else:
        #     self._graph.nodes[node]["log_r"] = np.add(log_p, log_s, order="C")

        self._graph[node_idx].log_r = np.add(log_p, log_s, out=self._graph[node_idx].log_r, order="C")


class TreeNode(object):
    __slots__ = ("log_p", "log_r", "node_id")

    # log_p: np.array
    # log_r: np.array
    # node_id: Union[str | int]

    def __init__(self, log_p: np.array, log_r: np.array, node_id: Union[str | int]):
        self.log_p = log_p
        self.log_r = log_r
        self.node_id = node_id

    def __copy__(self):
        return TreeNode(np.copy(self.log_p, order='C'),
                        np.copy(self.log_r, order='C'),
                        self.node_id)


class PostOrderNodeUpdater(DFSVisitor):
    __slots__ = "tree"

    def __init__(self, tree):
        self.tree = tree

    def finish_vertex(self, v, t):
        node_id = self.tree._node_indices_rev[v]
        self.tree._update_node(node_id)


class GraphToDictVisitor(DFSVisitor):
    __slots__ = ("tree", "dict_of_dicts")

    def __init__(self, tree):
        self.tree = tree
        self.dict_of_dicts = defaultdict(dict)

    def tree_edge(self, edge):
        parent = edge[0]
        child = edge[1]

        parent_idx = self.tree._node_indices_rev[parent]
        child_idx = self.tree._node_indices_rev[child]

        self.dict_of_dicts[parent_idx][child_idx] = {}
        self.dict_of_dicts[child_idx] = {}


class PreOrderNodeRelabeller(DFSVisitor):
    __slots__ = ("tree", "data", "node_indices", "node_indices_rev", "orig_node_indices_rev", "graph", "orig_data")

    def __init__(self, tree, data, start_idx=0):
        self.tree = tree
        self.data = data
        self.orig_data = tree._data
        self.node_indices = dict()
        self.node_indices_rev = dict()
        self.orig_node_indices_rev = tree._node_indices_rev
        self.curr_idx = start_idx
        self.graph = tree._graph

    def discover_vertex(self, v, t):
        node_id = self.orig_node_indices_rev[v]
        if node_id != "root":
            data_listing = self.orig_data[node_id]
            node_id = self.curr_idx
            self.curr_idx += 1
            self.graph[v].node_id = node_id
            self.data[node_id] = data_listing

        self.node_indices[node_id] = v
        self.node_indices_rev[v] = node_id

# class GraphToDictVisitor(DFSVisitor):
#     def __init__(self, tree):
#         self.tree = tree
#         self.dict_of_dicts = defaultdict(dict)
#
#     def tree_edge(self, edge):
#         parent = edge[0]
#         child = edge[1]
#
#         parent_idx = self.tree._node_indices_rev[parent]
#         child_idx = self.tree._node_indices_rev[child]
#
#         self.dict_of_dicts[parent_idx][child_idx] = self.dict_of_dicts[child_idx]
