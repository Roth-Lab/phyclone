import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from phyclone.tree.utils import get_clades, compute_log_S
from phyclone.utils import get_iterator_length
from phyclone.utils.math import log_factorial


class OldTree(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = nx.DiGraph()

        self._add_node("root")

    def __hash__(self):
        return hash((get_clades(self), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (get_clades(self), frozenset(self.outliers))

        other_key = (get_clades(other), frozenset(other.outliers))

        return self_key == other_key

    @staticmethod
    def get_single_node_tree(data):
        """Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        tree = OldTree(data[0].grid_size)

        node = tree.create_root_node([])

        for data_point in data:
            tree._internal_add_data_point_to_node(True, data_point, node)

        tree.update()

        return tree

    @property
    def root_node_name(self):
        return "root"

    @property
    def outlier_node_name(self):
        return -1

    @property
    def graph(self):
        result = self._graph.copy()

        result.remove_node("root")

        return result

    @property
    def data(self):
        result = sorted(itertools.chain.from_iterable(self._data.values()), key=lambda x: x.idx)
        return result

    @property
    def data_log_likelihood(self):
        """The log likelihood grid of the data for all values of the root node."""
        return self._graph.nodes["root"]["log_R"]

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
        result = list(self._graph.nodes())

        result.remove("root")

        return result

    def get_number_of_nodes(self):
        return self._graph.number_of_nodes() - 1

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
        return list(self._graph.successors("root"))

    @staticmethod
    def from_dict(data, tree_dict):
        new = OldTree(data[0].grid_size)

        new._graph = nx.DiGraph(tree_dict["graph"])

        data = dict(zip([x.idx for x in data], data))

        for node in new._graph.nodes:
            new._add_node(node)

        for idx, node in tree_dict["labels"].items():
            new._internal_add_data_point_to_node(True, data[idx], node)

        new.update()

        return new

    def to_dict(self):
        return {"graph": nx.to_dict_of_dicts(self._graph), "labels": self.labels}

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._internal_add_data_point_to_node(False, data_point, node)

    def _internal_add_data_point_to_node(self, build_add, data_point, node):
        self._data[node].append(data_point)
        if node != -1:
            self._graph.nodes[node]["log_p"] += data_point.value

            self._graph.nodes[node]["log_R"] += data_point.value

            if not build_add:
                self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)

    def add_subtree(self, subtree, parent=None):
        first_label = (
            max(
                self.nodes
                + subtree.nodes
                + [
                    -1,
                ]
            )
            + 1
        )

        node_map = {}

        subtree = subtree.copy()

        for new_node, old_node in enumerate(subtree.nodes, first_label):
            node_map[old_node] = new_node

            self._data[new_node] = subtree._data[old_node]

        nx.relabel_nodes(subtree._graph, node_map, copy=False)

        self._graph = nx.compose(self._graph, subtree.graph)

        # Connect subtree
        if parent is None:
            parent = "root"

        for node in subtree.roots:
            self._graph.add_edge(parent, node)

        self._update_path_to_root(parent)

    def create_root_node(self, children=[], data=[]):
        """Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        data: list
            Data points to add to new node.
        """
        node = nx.number_of_nodes(self._graph) - 1

        self._add_node(node)

        for data_point in data:
            self._data[node].append(data_point)

            self._graph.nodes[node]["log_p"] += data_point.value

        self._graph.add_edge("root", node)

        for child in children:
            self._graph.remove_edge("root", child)

            self._graph.add_edge(node, child)

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

        for node in new._graph:
            new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()

            new._graph.nodes[node]["log_R"] = self._graph.nodes[node]["log_R"].copy()

        return new

    def get_children(self, node):
        return list(self._graph.successors(node))

    def get_number_of_children(self, node):
        children = self._graph.successors(node)
        num_children = get_iterator_length(children)
        return num_children

    def get_descendants(self, source="root"):
        return nx.descendants(self._graph, source=source)

    def get_number_of_descendants(self, source="root"):
        return len(nx.descendants(self._graph, source=source))

    def get_parent(self, node):
        if node == "root":
            return None

        else:
            return list(self._graph.predecessors(node))[0]

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

        new = OldTree(self.grid_size)

        subtree_graph = nx.dfs_tree(self._graph, subtree_root)

        new._graph = nx.compose(new._graph, subtree_graph)

        new._graph.add_edge("root", subtree_root)

        for node in new.nodes:
            new._data[node] = list(self._data[node])

            new._graph.nodes[node]["log_p"] = self._graph.nodes[node]["log_p"].copy()

        new.update()

        return new

    def get_subtree_data(self, node):
        data = self.get_data(node)

        for desc in self.get_descendants(node):
            data.extend(self.get_data(desc))

        return data

    def relabel_nodes(self):
        node_map = {}

        data = defaultdict(list)

        data[-1] = self._data[-1]

        new_node = 0

        for old_node in nx.dfs_preorder_nodes(self._graph, source="root"):
            if old_node == "root":
                continue

            node_map[old_node] = new_node

            data[new_node] = self._data[old_node]

            new_node += 1

        self._data = data

        self._graph = nx.relabel_nodes(self._graph, node_map)

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        if node != -1:
            self._graph.nodes[node]["log_p"] -= data_point.value

            self._update_path_to_root(node)

    def remove_data_point_from_outliers(self, data_point):
        self._data[-1].remove(data_point)

    def remove_subtree(self, subtree):
        if subtree == self:
            self.__init__(self.grid_size)

        else:
            assert len(subtree.roots) == 1

            parent = self.get_parent(subtree.roots[0])

            self._graph.remove_nodes_from(subtree.nodes)

            for node in subtree.nodes:
                del self._data[node]

            self._update_path_to_root(parent)

    def update(self):
        for node in nx.dfs_postorder_nodes(self._graph, "root"):
            self._update_node(node)

    def _add_node(self, node):
        self._graph.add_node(node)
        self._graph.nodes[node]["log_p"] = np.full(self.grid_size, self._log_prior, order="C")
        self._graph.nodes[node]["log_R"] = np.zeros(self.grid_size, order="C")

    def _update_path_to_root(self, source):
        """Update recursion values for all nodes on the path between the source node and root inclusive."""
        paths = list(nx.all_simple_paths(self._graph, "root", source))

        if len(paths) == 0:
            assert source == "root"

            paths = [["root"]]

        assert len(paths) == 1

        path = paths[0]

        assert path[-1] == source

        assert path[0] == "root"

        for source in reversed(path):
            self._update_node(source)

    def _update_node(self, node):
        child_log_r_values = [self._graph.nodes[child]["log_R"] for child in self._graph.successors(node)]

        log_p = self._graph.nodes[node]["log_p"]

        if len(child_log_r_values) == 0:
            self._graph.nodes[node]["log_R"] = log_p.copy()
            return
        else:
            log_s = compute_log_S(child_log_r_values)

        if "log_R" in self._graph.nodes[node]:
            self._graph.nodes[node]["log_R"] = np.add(log_p, log_s, out=self._graph.nodes[node]["log_R"], order="C")
        else:
            self._graph.nodes[node]["log_R"] = np.add(log_p, log_s, order="C")
