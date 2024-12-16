from collections import defaultdict
from rustworkx.visit import DFSVisitor


class PostOrderNodeUpdater(DFSVisitor):
    __slots__ = "node_update_fxn"

    def __init__(self, node_update_fxn):
        self.node_update_fxn = node_update_fxn

    def finish_vertex(self, v, t):
        self.node_update_fxn(v)


class PreOrderNodeRelabeller(DFSVisitor):
    __slots__ = (
        "data",
        "node_indices",
        "node_indices_rev",
        "graph",
        "orig_data",
        "curr_idx",
    )

    def __init__(self, tree, data, start_idx=0):
        self.data = data
        self.orig_data = tree._data
        self.node_indices = dict()
        self.node_indices_rev = dict()
        self.curr_idx = start_idx
        self.graph = tree._graph

    def discover_vertex(self, v, t):
        node_id = self.graph[v].node_id
        if node_id != "root":
            old_node_id = node_id
            node_id = self.curr_idx
            self.curr_idx += 1
            self.graph[v].node_id = node_id
            self.data[node_id] = self.orig_data[old_node_id]

        self.node_indices[node_id] = v
        self.node_indices_rev[v] = node_id


class GraphToCladesVisitor(DFSVisitor):
    __slots__ = (
        "dict_of_sets",
        "child_parent_mapping",
        "clades",
        "node_indices_rev",
        "data",
    )

    def __init__(self, tree):
        self.dict_of_sets = defaultdict(set)
        self.child_parent_mapping = dict()
        self.clades = set()
        self.node_indices_rev = tree._node_indices_rev
        self.data = tree._data

    def discover_vertex(self, v, t):
        node_idx = self.node_indices_rev[v]

        datalist = self.data[node_idx]

        mut_set = {dp.idx for dp in datalist}
        self.dict_of_sets[node_idx] = mut_set

    def tree_edge(self, edge):
        parent = edge[0]
        child = edge[1]

        parent_idx = self.node_indices_rev[parent]
        child_idx = self.node_indices_rev[child]
        self.child_parent_mapping[child_idx] = parent_idx

    def finish_vertex(self, v, t):
        node_idx = self.node_indices_rev[v]

        if node_idx != "root":
            parent_idx = self.child_parent_mapping[node_idx]
            datalist = self.dict_of_sets[node_idx]

            self.dict_of_sets[parent_idx].update(datalist)
            self.clades.add(frozenset(datalist))


class GraphToNewickVisitor(DFSVisitor):
    __slots__ = (
        "dict_of_lists",
        "child_parent_mapping",
        "parents",
        "node_indices_rev",
        "final_string",
    )

    def __init__(self, tree):
        self.dict_of_lists = defaultdict(list)
        self.child_parent_mapping = dict()
        self.parents = set()
        self.node_indices_rev = tree._node_indices_rev
        self.final_string = None

    def tree_edge(self, edge):
        parent = edge[0]
        child = edge[1]

        parent_idx = self.node_indices_rev[parent]
        child_idx = self.node_indices_rev[child]

        self.child_parent_mapping[child_idx] = parent_idx
        self.parents.add(parent_idx)

    def finish_vertex(self, v, t):
        node_idx = self.node_indices_rev[v]

        if node_idx in self.parents:
            curr_list = self.dict_of_lists[node_idx]
            child_strings = ",".join(curr_list)
            curr_node_string = "({child_strings}){node_idx}".format(child_strings=child_strings, node_idx=node_idx)
        else:
            curr_node_string = "{node_idx}".format(node_idx=node_idx)

        if node_idx != "root":
            parent_idx = self.child_parent_mapping[node_idx]
            self.dict_of_lists[parent_idx].append(curr_node_string)
        else:
            self.final_string = curr_node_string + ";"
