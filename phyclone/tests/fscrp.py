from collections import Counter, defaultdict
import networkx as nx
import numpy as np


def add_clonal_prev(dim, graph, rng):
    clonal_prevs = rng.dirichlet(np.ones(graph.number_of_nodes()), size=dim)

    for i, node in enumerate(graph.nodes):
        graph.nodes[node]["clonal_prev"] = clonal_prevs[:, i]


def add_cellular_prev(graph):
    for node in nx.dfs_postorder_nodes(graph):
        children = list(graph.successors(node))

        graph.nodes[node]["cellular_prev"] = graph.nodes[node]["clonal_prev"].copy()

        graph.nodes[node]["cellular_prev"] += np.sum(
            np.array([graph.nodes[c]["cellular_prev"] for c in children]),
            axis=0,
        )


def simulate_fscrp_tree(rng, alpha=1.0, dim=1, forest=False, num_data_points=20):
    """Simulate an FSCRP tree.

    Parameters
    ----------
    rng: np.random.Generator
    alpha: float
        Concentration param for CRP
    dim: int
        Number of dimensions of the data
    forest: bool
        Whether to simulate a forest i.e. multiple trees
    num_data_points: int
        Number of data points

    Returns
    -------
    G: nx.DiGraph
        Graph representing tree (forest) with emission parameters set at  nodes.
    """

    labels = simulate_labels(num_data_points, rng, alpha=alpha)

    tree = simulate_tree(labels, rng, forest=forest)

    # simulate_params(tree, dim=dim)
    add_clonal_prev(dim, tree, rng)
    add_cellular_prev(tree)

    return tree


def get_roots(G):
    """Find all root nodes in the graph."""
    roots = []

    for node in G.nodes():
        if G.in_degree(node) == 0:
            roots.append(node)

    return roots


def simulate_labels(num_data_points, rng, alpha=1.0):
    """Simulate clustering from CRP.

    Parameters
    ----------
    num_data_points: int
        Number of data points to assign to clusters.
    alpha: float
        CRP concentration paramter

    Returns
    -------
    labels: list (int)
        List of cluster labels.
    """
    labels = [
        0,
    ]

    for _ in range(1, num_data_points):
        counts = Counter(labels)

        probs = list(counts.values())

        probs.append(alpha)

        probs = np.array(probs)

        probs = probs / np.sum(probs)

        labels.append(rng.multinomial(1, probs).argmax())

    return labels


def simulate_tree(labels, rng, forest=False):
    """Simulate a FSCRP tree.

    Parameters
    ----------
    labels: list (int)
        List of labels for data points.

    Returns
    -------
    G: nx.DiGraph
        Graph representing tree (forest).
    """
    G = nx.DiGraph()

    num_nodes = len(set(labels))

    node_label_map = defaultdict(list)

    for i, l in enumerate(labels):
        node_label_map[l].append("m{}".format(i))

    nodes_sorted = sorted(node_label_map.items(), key=lambda x: len(x[1]))

    G.add_node(nodes_sorted[0][0])

    for node in range(1, num_nodes):
        roots = get_roots(G)

        if (not forest) and (node == num_nodes - 1):
            num_children = len(roots)

        else:
            num_children = rng.integers(0, len(roots), endpoint=True)

        children = rng.choice(roots, num_children, replace=False)

        node_to_add = nodes_sorted[node][0]

        G.add_node(node_to_add)

        for child in children:
            G.add_edge(node_to_add, child)

    for node in G.nodes:
        G.nodes[node]["snvs"] = node_label_map[node]

    roots = get_roots(G)
    assert len(roots) == 1
    G.graph["root"] = roots[0]

    return G


def simulate_params(tree, rng, dim=1, kappa=1.0):
    """Simulate cluster parameters.

    Parameters
    ----------
    tree: nx.DiGraph
        Graph representing tree (forest).
    kappa: float
        Dirichlet distribution parameter for node parameters.

    Returns
    -------
    params: dict
        Mapping of nodes to aggregate (over subtree rooted at node) parameters.
    """

    def get_agg_params(node_params, tree, node=None):
        if node is None:
            agg_params = {}

            for node in get_roots(tree):
                agg_params.update(get_agg_params(node_params, tree, node=node))

        else:
            agg_params = {node: node_params[node]}

            for child in tree.successors(node):
                agg_params.update(get_agg_params(node_params, tree, node=child))

                agg_params[node] += agg_params[child]

        return agg_params

    node_params = rng.dirichlet(np.ones(len(tree.nodes)) * kappa, size=dim)

    node_params = dict(zip(tree.nodes, node_params.T))

    params = get_agg_params(node_params, tree)

    for node in tree.nodes:
        tree.nodes[node]["clonal_prev"] = node_params[node]

        tree.nodes[node]["cellular_prev"] = params[node]

    return params
