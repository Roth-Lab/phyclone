from collections import defaultdict
from scipy import fft

import networkx as nx
import numpy as np

from phyclone.consensus import get_clades
from phyclone.math_utils import log_sum_exp
import itertools
from phyclone.utils import list_of_np_cache


class FSCRPDistribution(object):
    """ FSCRP prior distribution on trees.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, tree):
        log_p = 0

        # CRP prior
        num_nodes = len(tree.nodes)

        log_p += num_nodes * np.log(self.alpha)

        log_factorial_sum = tree.log_factorial_sum

        log_p += log_factorial_sum

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        return log_p


class TreeJointDistribution(object):

    def __init__(self, prior):
        self.prior = prior

    def log_p(self, tree):
        """ The log likelihood of the data marginalized over root node parameters.
        """
        log_p = self.prior.log_p(tree)

        log_p_res = self._get_log_p_precomputed_vals_from_tree(tree)

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += log_sum_exp(tree.data_log_likelihood[i, :])

        log_p += log_p_res

        return log_p

    def _get_log_p_precomputed_vals_from_tree(self, tree):
        log_p_outlier_probs_out_node = tree.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_outlier_node']
        log_p_outlier_probs_in_nodes = tree.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_included_nodes']
        log_p_outlier_marginal_probs = tree.sum_of_outlier_data_points_marginal_prob
        log_p_res = log_p_outlier_probs_out_node + log_p_outlier_probs_in_nodes + log_p_outlier_marginal_probs
        return log_p_res

    def log_p_one(self, tree):
        """ The log likelihood of the data conditioned on the root having value 1.0 in all dimensions.
        """
        log_p = self.prior.log_p(tree)

        log_p_res = self._get_log_p_precomputed_vals_from_tree(tree)

        if len(tree.roots) > 0:
            for i in range(tree.grid_size[0]):
                log_p += tree.data_log_likelihood[i, -1]

        log_p += log_p_res

        return log_p


def get_set_hash(datapoints_set):
    ret = hash(frozenset(datapoints_set))
    return ret


class Tree(object):

    def __init__(self, grid_size, factorial_arr, memo_logs):
        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = nx.DiGraph()

        self._add_node("root")

        self.sum_of_log_data_points_outlier_prob_gt_zero = {'data_points_on_outlier_node': 0,
                                                            'data_points_on_included_nodes': 0}

        self.sum_of_outlier_data_points_marginal_prob = 0

        self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise = defaultdict(int)

        self.factorial_arr = factorial_arr

        self.log_p_comp_memo = memo_logs["log_p"]

        self.memo_logs = memo_logs

        self.log_factorial_sum = 0

        self.log_factorials_nodewise = defaultdict(int)

        tmp_hash = get_set_hash({"log_p"})
        tmp_hash_2 = get_set_hash({"zeros"})
        if tmp_hash not in self.log_p_comp_memo:
            self.log_p_comp_memo[tmp_hash] = np.ones(self.grid_size) * self._log_prior
        if tmp_hash_2 not in self.log_p_comp_memo:
            self.log_p_comp_memo[tmp_hash_2] = np.zeros(self.grid_size)
            # self.log_r_comp_memo[tmp_hash_2] = self.log_p_comp_memo[tmp_hash_2]

    def __hash__(self):
        return hash((get_clades(self), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (get_clades(self), frozenset(self.outliers))

        other_key = (get_clades(other), frozenset(other.outliers))

        return self_key == other_key

    @staticmethod
    def get_single_node_tree(data, factorial_arr, memo_logs):
        """ Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        tree = Tree(data[0].grid_size, factorial_arr, memo_logs)

        node = tree.create_root_node([])

        for data_point in data:
            tree.add_data_point_to_node(data_point, node)

        return tree

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
        """ The log likelihood grid of the data for all values of the root node.
        """
        return self._graph.nodes["root"]["log_R"]
        # return self.log_r_comp_memo[self._graph.nodes["root"]["datapoints_log_R"]]

    @property
    def labels(self):

        result = {dp.idx: k for k, l in self.node_data.items() for dp in l}

        return result

    @property
    def nodes(self):
        result = list(self._graph.nodes())

        result.remove("root")

        return result

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
        new = Tree(data[0].grid_size, None)  # TODO: check with Andy if this needs to be able to compute stuff

        new._graph = nx.DiGraph(tree_dict["graph"])

        data = dict(zip([x.idx for x in data], data))

        for node in new._graph.nodes:
            new._add_node(node)

        for idx, node in tree_dict["labels"].items():
            new._data[node].append(data[idx])

            if node != -1:
                new._graph.nodes[node]["log_p"] += data[idx].value

                new._graph.nodes[node]["log_R"] += data[idx].value

        new.update()

        return new

    def to_dict(self):
        return {
            "graph": nx.to_dict_of_dicts(self._graph),
            "labels": self.labels
        }

    def add_item_to_dp_set_update_global(self, data_point, edit_set):
        old_set_hash = get_set_hash(edit_set)
        edit_set.add(data_point.name)
        set_hash = get_set_hash(edit_set)
        if set_hash not in self.log_p_comp_memo:
            base_val = self.log_p_comp_memo[old_set_hash]
            self.log_p_comp_memo[set_hash] = base_val + data_point.value
        return set_hash

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._add_datapoint_to_log_factorial_trackers(node, len(self._data[node]))

        self._data[node].append(data_point)

        if data_point.outlier_prob > 0:
            self._add_datapoint_to_log_val_trackers(data_point, node)

        if node != -1:
            self._graph.nodes[node]["log_R"] += data_point.value
            _ = self.add_item_to_dp_set_update_global(data_point, self._graph.nodes[node]["datapoints_log_p"])

            self._update_path_to_root(self.get_parent(node))

    def _add_datapoint_to_log_val_trackers(self, data_point, node):
        if data_point.outlier_prob > 0:
            log_p_val_adjust = np.log1p(-data_point.outlier_prob)

            self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_included_nodes'] += log_p_val_adjust
            self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] += log_p_val_adjust

    def _add_datapoint_to_outlier_log_val_trackers(self, data_point, node):
        if data_point.outlier_prob > 0:
            log_p_val_adjust = np.log(data_point.outlier_prob)

            self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_outlier_node'] += log_p_val_adjust
            self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] += log_p_val_adjust

        self.sum_of_outlier_data_points_marginal_prob += data_point.outlier_marginal_prob

    def _remove_datapoint_from_log_val_trackers(self, data_point, node):
        if data_point.outlier_prob > 0:
            log_p_val_adjust = np.log1p(-data_point.outlier_prob)

            self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_included_nodes'] -= log_p_val_adjust
            self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] -= log_p_val_adjust

    def _remove_datapoint_from_outlier_log_val_trackers(self, data_point, node):
        if data_point.outlier_prob > 0:
            log_p_val_adjust = np.log(data_point.outlier_prob)

            self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_outlier_node'] -= log_p_val_adjust
            self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] -= log_p_val_adjust

        self.sum_of_outlier_data_points_marginal_prob -= data_point.outlier_marginal_prob

    def _add_node_to_log_val_trackers(self,
                                      outlier_prob_gt_zero_node_val,
                                      node):

        self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_included_nodes'] \
            += outlier_prob_gt_zero_node_val
        self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] = outlier_prob_gt_zero_node_val

    def _add_node_to_outlier_log_val_trackers(self,
                                              outlier_prob_gt_zero_node_val,
                                              outlier_data_points_marginal_prob_node_val,
                                              node):

        self.sum_of_outlier_data_points_marginal_prob = outlier_data_points_marginal_prob_node_val

        self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_outlier_node'] += outlier_prob_gt_zero_node_val
        self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] = outlier_prob_gt_zero_node_val

    def _remove_node_from_log_val_trackers(self,
                                           outlier_prob_gt_zero_node_val,
                                           node):

        self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_included_nodes'] \
            -= outlier_prob_gt_zero_node_val
        # self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise.pop(node, None)
        self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] = 0

    def _remove_node_from_outlier_log_val_trackers(self,
                                                   outlier_prob_gt_zero_node_val,
                                                   node):

        self.sum_of_outlier_data_points_marginal_prob = 0

        self.sum_of_log_data_points_outlier_prob_gt_zero['data_points_on_outlier_node'] -= outlier_prob_gt_zero_node_val
        self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node] = 0

    def _add_datapoint_to_log_factorial_trackers(self, node, old_node_data_length):
        if node == -1:
            return

        old_node_log_factorial_val = self.log_factorials_nodewise[node]

        new_node_data_length = old_node_data_length + 1

        new_node_log_factorial_val = self.factorial_arr[new_node_data_length - 1]

        self.log_factorial_sum -= old_node_log_factorial_val
        self.log_factorial_sum += new_node_log_factorial_val
        self.log_factorials_nodewise[node] = new_node_log_factorial_val

    def _remove_datapoint_from_log_factorial_trackers(self, node, old_node_data_length):
        if node == -1:
            return

        old_node_log_factorial_val = self.log_factorials_nodewise[node]

        new_node_data_length = old_node_data_length - 1

        new_node_log_factorial_val = self.factorial_arr[new_node_data_length - 1]

        self.log_factorial_sum -= old_node_log_factorial_val
        self.log_factorial_sum += new_node_log_factorial_val
        self.log_factorials_nodewise[node] = new_node_log_factorial_val

    def _add_node_to_log_factorial_trackers(self, node, node_data_length):
        if node == -1:
            return

        node_log_factorial_val = self.factorial_arr[node_data_length - 1]

        self.log_factorial_sum += node_log_factorial_val
        self.log_factorials_nodewise[node] = node_log_factorial_val

    def _remove_node_from_log_factorial_trackers(self, node, node_data_length):
        if node == -1:
            return

        node_log_factorial_val = self.factorial_arr[node_data_length - 1]

        self.log_factorial_sum -= node_log_factorial_val
        self.log_factorials_nodewise[node] = 0

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)
        self._add_datapoint_to_outlier_log_val_trackers(data_point, -1)

    def add_subtree(self, subtree, parent=None):
        first_label = max(self.nodes + subtree.nodes + [-1, ]) + 1

        node_map = {}

        subtree = subtree.copy()

        for new_node, old_node in enumerate(subtree.nodes, first_label):
            node_map[old_node] = new_node

            node_data = subtree._data[old_node]

            self._data[new_node] = node_data

            outlier_prob_gt_zero_node_val = subtree.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[old_node]

            self._add_node_to_log_val_trackers(outlier_prob_gt_zero_node_val, new_node)

            self._add_node_to_log_factorial_trackers(new_node, len(node_data))

        nx.relabel_nodes(subtree._graph, node_map, copy=False)

        self._graph = nx.compose(self._graph, subtree.graph)

        # Connect subtree
        if parent is None:
            parent = "root"

        for node in subtree.roots:
            self._graph.add_edge(parent, node)

        self._update_path_to_root(parent)

    def create_root_node(self, children=[], data=[]):
        """ Create a new root node in the forest.

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

            if data_point.outlier_prob > 0:
                self._add_datapoint_to_log_val_trackers(data_point, node)

            self.add_item_to_dp_set_update_global(data_point, self._graph.nodes[node]["datapoints_log_p"])

        self._add_node_to_log_factorial_trackers(node, len(data))

        self._graph.add_edge("root", node)

        for child in children:
            self._graph.remove_edge("root", child)

            self._graph.add_edge(node, child)

        self._update_path_to_root(node)

        return node

    def copy(self):
        memo_logs = self.memo_logs

        cls = self.__class__

        new = cls.__new__(cls)

        new.grid_size = self.grid_size

        new._data = defaultdict(list)

        new.sum_of_log_data_points_outlier_prob_gt_zero = {'data_points_on_outlier_node': 0,
                                                           'data_points_on_included_nodes': 0}

        new.sum_of_outlier_data_points_marginal_prob = 0

        new.sum_of_log_data_points_outlier_prob_gt_zero_nodewise = defaultdict(int)

        new.factorial_arr = self.factorial_arr

        new.memo_logs = memo_logs

        new.log_p_comp_memo = memo_logs["log_p"]

        new.log_factorial_sum = 0

        new.log_factorials_nodewise = defaultdict(int)

        for node in self._data:
            node_data = list(self._data[node])
            new._data[node] = node_data

            outlier_prob_gt_zero_node_val = self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node]

            if node == -1:
                outlier_data_points_marginal_prob_node_val = self.sum_of_outlier_data_points_marginal_prob
                new._add_node_to_outlier_log_val_trackers(outlier_prob_gt_zero_node_val,
                                                          outlier_data_points_marginal_prob_node_val,
                                                          node)
            else:
                new._add_node_to_log_val_trackers(outlier_prob_gt_zero_node_val, node)
                new._add_node_to_log_factorial_trackers(node, len(node_data))

        new._log_prior = self._log_prior

        new._graph = self._graph.copy()

        for node in new._graph:

            new._graph.nodes[node]["log_R"] = self._graph.nodes[node]["log_R"].copy()
            new._graph.nodes[node]["datapoints_log_p"] = self._graph.nodes[node]["datapoints_log_p"].copy()

        return new

    def get_children(self, node):
        return list(self._graph.successors(node))

    def get_descendants(self, source="root"):
        return nx.descendants(self._graph, source=source)

    def get_parent(self, node):
        if node == "root":
            return None

        else:
            return list(self._graph.predecessors(node))[0]

    def get_data(self, node):
        return list(self._data[node])

    def get_subtree(self, subtree_root):
        if subtree_root == "root":
            return self.copy()

        new = Tree(self.grid_size, self.factorial_arr, self.memo_logs)

        subtree_graph = nx.dfs_tree(self._graph, subtree_root)

        new._graph = nx.compose(new._graph, subtree_graph)

        new._graph.add_edge("root", subtree_root)

        for node in new.nodes:
            node_data = list(self._data[node])
            new._data[node] = node_data

            new._graph.nodes[node]["datapoints_log_p"] = self._graph.nodes[node]["datapoints_log_p"].copy()

            outlier_prob_gt_zero_node_val = self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node]

            if node == -1:
                outlier_data_points_marginal_prob_node_val = self.sum_of_outlier_data_points_marginal_prob
                new._add_node_to_outlier_log_val_trackers(outlier_prob_gt_zero_node_val,
                                                          outlier_data_points_marginal_prob_node_val,
                                                          node)
            else:
                new._add_node_to_log_val_trackers(outlier_prob_gt_zero_node_val, node)
                new._add_node_to_log_factorial_trackers(node, len(node_data))

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
        self._remove_datapoint_from_log_factorial_trackers(node, len(self._data[node]))

        self._data[node].remove(data_point)

        if data_point.outlier_prob > 0:
            self._remove_datapoint_from_log_val_trackers(data_point, node)

        if node != -1:
            self._graph.nodes[node]["datapoints_log_p"].remove(data_point.name)

            self._update_path_to_root(node)

    def remove_data_point_from_outliers(self, data_point):
        self._data[-1].remove(data_point)
        self._remove_datapoint_from_outlier_log_val_trackers(data_point, -1)

    def remove_subtree(self, subtree):
        if subtree == self:
            self.__init__(self.grid_size, self.factorial_arr, self.memo_logs)

        else:
            assert len(subtree.roots) == 1

            parent = self.get_parent(subtree.roots[0])

            self._graph.remove_nodes_from(subtree.nodes)

            for node in subtree.nodes:
                outlier_prob_gt_zero_node_val = self.sum_of_log_data_points_outlier_prob_gt_zero_nodewise[node]
                if node == -1:
                    self._remove_node_from_outlier_log_val_trackers(outlier_prob_gt_zero_node_val, node)
                else:
                    self._remove_node_from_log_val_trackers(outlier_prob_gt_zero_node_val, node)
                    self._remove_node_from_log_factorial_trackers(node, len(self._data[node]))

                del self._data[node]

            self._update_path_to_root(parent)

    def update(self):
        for node in nx.dfs_postorder_nodes(self._graph, "root"):
            self._update_node(node)

    def _add_node(self, node):
        self._graph.add_node(node)
        self._graph.nodes[node]["log_R"] = np.zeros(self.grid_size)
        self._graph.nodes[node]["datapoints_log_p"] = {"log_p"}

    def _update_path_to_root(self, source):
        """ Update recursion values for all nodes on the path between the source node and root inclusive.
        """
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

        child_log_R_values = [self._graph.nodes[child]["log_R"] for child in self._graph.successors(node)]

        log_s = compute_log_S(child_log_R_values)

        if isinstance(log_s, float):
            log_s = np.zeros(self.grid_size)

        log_p_set_hash = get_set_hash(self._graph.nodes[node]["datapoints_log_p"])

        self._graph.nodes[node]["log_R"] = np.add(self.log_p_comp_memo[log_p_set_hash], log_s, order='C')


@list_of_np_cache(maxsize=1024)
def compute_log_S(child_log_R_values):
    """ Compute log(S) recursion.

    Parameters
    ----------
    child_log_R_values: ndarray
        log_R values from child nodes.
    """
    if len(child_log_R_values) == 0:
        return 0.0

    log_D = compute_log_D(child_log_R_values)

    log_S = np.zeros(log_D.shape)

    num_dims = log_D.shape[0]

    for i in range(num_dims):
        log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])

    return log_S


def compute_log_D(child_log_R_values):
    if len(child_log_R_values) == 0:
        return 0

    fft_log_d = _comp_log_d_fft(child_log_R_values)

    log_D = fft_log_d

    return log_D


def _comp_log_d_split(child_log_R_values):
    num_children = len(child_log_R_values)
    if num_children == 1:
        return child_log_R_values[0].copy()

    log_D = child_log_R_values[0].copy()
    num_dims = log_D.shape[0]
    num_children = child_log_R_values.shape[0]

    for j in range(1, num_children):
        child_log_R = child_log_R_values[j]
        for i in range(num_dims):
            log_D[i, :] = _compute_log_D_n(child_log_R[i, :], log_D[i, :])
    return log_D


def _comp_log_d_fft(child_log_R_values):
    num_children = len(child_log_R_values)

    if num_children == 1:
        return child_log_R_values[0].copy()

    maxes = np.max(child_log_R_values, axis=-1, keepdims=True)
    child_log_R_values_norm = np.expm1(child_log_R_values - maxes)

    relevant_axis_length = child_log_R_values.shape[-1]

    outlen = relevant_axis_length + relevant_axis_length - 1

    pad_to = fft.next_fast_len(outlen, real=True)

    fwd = fft.rfft(child_log_R_values_norm, n=pad_to, axis=-1)

    c_fft = fwd * fwd

    log_D = fft.irfft(c_fft, n=pad_to, axis=-1)

    log_D = log_D[..., :relevant_axis_length]

    log_D = np.log1p(log_D) + maxes
    log_D = np.add.reduce(log_D)

    return log_D


def _compute_log_D_n(child_log_R, prev_log_D_n):
    """ Compute the recursion over D not using the FFT.
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.expm1(child_log_R - log_R_max)

    D_norm = np.expm1(prev_log_D_n - log_D_max)

    result = np.convolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    # result[result <= 0] = 1e-100

    return np.log1p(result) + log_D_max + log_R_max
