from collections import defaultdict
from scipy.special import logsumexp as log_sum_exp
from scipy.signal import fftconvolve

import copy
import networkx as nx
import numpy as np

from phyclone.consensus import get_clades
from phyclone.math_utils import log_factorial


class Tree(object):
    def __init__(self, alpha, grid_size):
        self.alpha = alpha

        self.grid_size = grid_size

        self._data = defaultdict(list)

        self._log_prior = -np.log(grid_size[1])

        self._graph = nx.DiGraph()

        self._add_node('root')

    def __hash__(self):
        return hash((self.alpha, get_clades(self), frozenset(self.outliers)))

    def __eq__(self, other):
        self_key = (self.alpha, get_clades(self), frozenset(self.outliers))

        other_key = (other.alpha, get_clades(other), frozenset(other.outliers))

        return self_key == other_key

    @staticmethod
    def get_single_node_tree(data):
        """ Load a tree with all data points assigned single node.

        Parameters
        ----------
        data: list
            Data points.
        """
        tree = Tree(1.0, data[0].grid_size)

        node = tree.create_root_node([])

        for data_point in data:
            tree.add_data_point_to_node(data_point, node)

        return tree

    @property
    def graph(self):
        result = self._graph.copy()

        result.remove_node('root')

        return result

    @property
    def data(self):
        result = []

        for node in self._data:
            result.extend(self._data[node])

        result = sorted(result, key=lambda x: x.idx)

        return result

    @property
    def data_log_likelihood(self):
        """ The log likelihood grid of the data for all values of the root node.
        """
        return self._graph.nodes['root']['log_R']

    @property
    def data_marginal_log_likelihood(self):
        """ The log likelihood of the data marginalized over root node parameters.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += log_sum_exp(self.data_log_likelihood[i, :])

        for data_point in self.outliers:
            log_p += np.sum(log_sum_exp(data_point.value + self._log_prior, axis=1))

        return log_p

    @property
    def data_conditional_log_likelihood(self):
        """ The log likelihood of the data conditioned on the root having value 1.0 in all dimensions.
        """
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += self.data_log_likelihood[i, -1]

        for data_point in self.outliers:
            log_p += np.sum(log_sum_exp(data_point.value + self._log_prior, axis=1))

        return log_p

    @property
    def labels(self):
        result = {}

        for node, node_data in self.node_data.items():
            for data_point in node_data:
                result[data_point.idx] = node

        return result

    @property
    def log_p(self):
        """ Log joint probability.
        """
        return self.log_p_prior + self.data_marginal_log_likelihood

    @property
    def log_p_one(self):
        """ The joint probability of the data conditioned on the root having value 1.0 in all dimensions.
        """
        return self.log_p_prior + self.data_conditional_log_likelihood

    @property
    def log_p_prior(self):
        """ Log prior from the FSCRP and outlier contributions.
        """
        log_p = 0

        # Outlier prior
        for node, node_data in self.node_data.items():
            for data_point in node_data:
                if node == -1:
                    log_p += np.log(data_point.outlier_prob)

                else:
                    log_p += np.log(1 - data_point.outlier_prob)

        # CRP prior
        num_nodes = nx.number_of_nodes(self._graph) - 1

        log_p += num_nodes * np.log(self.alpha)

        for node, node_data in self.node_data.items():
            if node == -1:
                continue

            num_data_points = len(node_data)

            log_p += log_factorial(num_data_points - 1)

        # Uniform prior on toplogies
        log_p -= (num_nodes - 1) * np.log(num_nodes + 1)

        return log_p

    @property
    def log_p_sigma(self):
        """ Log probability of the permutation.
        """
        # TODO: Check this. Are we missing a term for the outliers.
        # Correction for auxillary distribution
        log_p = 0

        for node in self._graph.nodes():
            children = list(self._graph.successors(node))

            log_p += log_factorial(sum([len(self._data[child]) for child in children]))

            for child in children:
                log_p -= log_factorial(len(self._data[child]))

            log_p += log_factorial(len(self._data[node]))

        return -log_p

    @property
    def nodes(self):
        result = list(self._graph.nodes())

        result.remove('root')

        return result

    @property
    def node_data(self):
        result = self._data.copy()

        if 'root' in result:
            del result['root']

        return result

    @property
    def outliers(self):
        return list(self._data[-1])

    @property
    def roots(self):
        return list(self._graph.successors('root'))

    def add_data_point_to_node(self, data_point, node):
        assert data_point.idx not in self.labels.keys()

        self._data[node].append(data_point)

        self._graph.nodes[node]['log_p'] += data_point.value

        self._graph.nodes[node]['log_R'] += data_point.value

        self._update_path_to_root(self.get_parent(node))

    def add_data_point_to_outliers(self, data_point):
        self._data[-1].append(data_point)

    def add_subtree(self, subtree, parent=None):
        first_label = max(self.nodes + subtree.nodes + [-1, ]) + 1

        node_map = {}

        subtree = subtree.copy()

        for new_node, old_node in enumerate(subtree.nodes, first_label):
            node_map[old_node] = new_node

            self._data[new_node] = subtree._data[old_node]

        nx.relabel_nodes(subtree._graph, node_map, copy=False)

        self._graph = nx.compose(self._graph, subtree.graph)

        # Connect subtree
        if parent is None:
            parent = 'root'

        for node in subtree.roots:
            self._graph.add_edge(parent, node)

        self._update_path_to_root(parent)

        self._relabel_nodes()

    def create_root_node(self, children=[]):
        """ Create a new root node in the forest.

        Parameters
        ----------
        children: list
            Children of the new node.
        """
        node = nx.number_of_nodes(self._graph) - 1

        self._add_node(node)

        self._graph.add_edge('root', node)

        for child in children:
            self._graph.remove_edge('root', child)

            self._graph.add_edge(node, child)

        self._update_path_to_root(node)

        return node

    def copy(self):
        cls = self.__class__

        new = cls.__new__(cls)

        new.alpha = self.alpha

        new.grid_size = self.grid_size

        new._data = defaultdict(list)

        for node in self._data:
            new._data[node] = list(self._data[node])

        new._log_prior = self._log_prior

        new._graph = copy.deepcopy(self._graph)

        return new

    def get_children(self, node):
        return list(self._graph.successors(node))

    def get_parent(self, node):
        if node == 'root':
            return None

        else:
            return list(self._graph.predecessors(node))[0]

    def get_data(self, node):
        return list(self._data[node])

    def get_subtree(self, subtree_root):
        if subtree_root == 'root':
            return self.copy()

        new = Tree(self.alpha, self.grid_size)

        subtree_graph = nx.dfs_tree(self._graph, subtree_root)

        new._graph = nx.compose(new._graph, subtree_graph)

        new._graph.add_edge('root', subtree_root)

        for node in new.nodes:
            new._data[node] = list(self._data[node])

            new._graph.nodes[node]['log_p'] = self._graph.nodes[node]['log_p'].copy()

        new.update()

        return new

    def remove_data_point_from_node(self, data_point, node):
        self._data[node].remove(data_point)

        self._graph.nodes[node]['log_p'] -= data_point.value

        self._update_path_to_root(node)

    def remove_data_point_from_outliers(self, data_point):
        self._data[-1].remove(data_point)

    def remove_subtree(self, subtree):
        if subtree == self:
            self.__init__(self.alpha, self.grid_size)

        else:
            assert len(subtree.roots) == 1

            parent = self.get_parent(subtree.roots[0])

            self._graph.remove_nodes_from(subtree.nodes)

            for node in subtree.nodes:
                del self._data[node]

            self._update_path_to_root(parent)

    def update(self):
        for node in nx.dfs_postorder_nodes(self._graph, 'root'):
            self._update_node(node)

    def _add_node(self, node):
        self._graph.add_node(node)

        self._graph.nodes[node]['log_p'] = np.ones(self.grid_size) * self._log_prior

        self._graph.nodes[node]['log_R'] = np.zeros(self.grid_size)

        self._graph.nodes[node]['log_S'] = np.zeros(self.grid_size)

    def _relabel_nodes(self):
        node_map = {}

        data = defaultdict(list)

        data[-1] = self._data[-1]

        for new_node, old_node in enumerate(self.nodes):
            node_map[old_node] = new_node

            data[new_node] = self._data[old_node]

        self._data = data

        self._graph = nx.relabel_nodes(self._graph, node_map)

    def _update_path_to_root(self, source):
        """ Update recursion values for all nodes on the path between the source node and root inclusive.
        """
        paths = list(nx.all_simple_paths(self._graph, 'root', source))

        if len(paths) == 0:
            assert source == 'root'

            paths = [['root']]

        assert len(paths) == 1

        path = paths[0]

        assert path[-1] == source

        assert path[0] == 'root'

        for source in reversed(path):
            self._update_node(source)

    def _update_node(self, node):
        child_log_R_values = [self._graph.nodes[child]['log_R'] for child in self._graph.successors(node)]

        self._graph.nodes[node]['log_S'] = compute_log_S(child_log_R_values)

        self._graph.nodes[node]['log_R'] = self._graph.nodes[node]['log_p'] + self._graph.nodes[node]['log_S']


def compute_log_S(child_log_R_values):
    """ Compute log(S) recursion.

    Parameters
    ----------
    child_log_R_values: list
        log_R values from child nodes.
    """
    if len(child_log_R_values) == 0:
        return 0

    log_D = compute_log_D(child_log_R_values)

    log_S = np.zeros(log_D.shape)

    num_dims = log_D.shape[0]

    for i in range(num_dims):
        log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])

    return log_S


def compute_log_D(child_log_R_values):
    if len(child_log_R_values) == 0:
        return 0

    log_D = child_log_R_values.pop(0).copy()

    num_dims = log_D.shape[0]

    for child_log_R in child_log_R_values:
        for i in range(num_dims):
            log_D[i, :] = _compute_log_D_n(child_log_R[i, :], log_D[i, :])

    return log_D


def _compute_log_D_n(child_log_R, prev_log_D_n):
    """ Compute the recursion over D using the FFT.
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = fftconvolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max
