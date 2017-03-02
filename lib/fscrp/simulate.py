from __future__ import division

from pydp.rvs import discrete_rvs

import random
import networkx as nx


def simulate_data(data_points, agg_func, node_param_sim_func, data_sim_func, alpha=1.0):
    '''
    Simulate a dataset usin the TSCRP.

    Args:
        data_points : (list) List of identifiers for each data point.

        agg_func : (function) Function for aggregating parameters from sub-tree. Should talk a list of parameters and
        return a parameter.

        node_param_sim_func : (function) Function simulation node parameters. Should take a list of lists representing
        the clustering and return a list of parameters of the same length.

        data_sim_func : (function) Function for simulation data for each data point. Should take the aggregated node
        parameter and return a data point.

        alpha : (float) Concentration parameter for CRP clustering.
    '''

    clusters = simulate_clusters(data_points, alpha)

    random.shuffle(clusters)

    forest = simulate_forest(len(clusters))

    node_params = node_param_sim_func(clusters)

    for i, node in enumerate(nx.dfs_preorder_nodes(forest)):
        forest.node[node]['node_params'] = node_params[i]

        forest.node[node]['data_points'] = clusters[i]

    for node in nx.dfs_postorder_nodes(forest):
        children = forest.successors(node)

        if len(children) == 0:
            forest.node[node]['agg_params'] = forest.node[node]['node_params']

        else:
            params = [forest.node[x]['node_params'] for x in nx.dfs_tree(forest, node)]

            forest.node[node]['agg_params'] = agg_func(params)

    data = {}

    for node in forest.nodes():
        node_data = data_sim_func(forest.node[node])

        for name, x in zip(forest.node[node]['data_points'], node_data):
            data[name] = x

    return data, forest


def simulate_clusters(data_points, alpha):
    counts = []

    clusters = []

    for name in data_points:
        norm_const = (sum(counts) + alpha)

        p = [x / norm_const for x in counts]

        p.append(alpha / norm_const)

        index = discrete_rvs(p)

        if index == len(clusters):
            counts.append(1)

            clusters.append([name, ])

        else:
            counts[index] += 1

            clusters[index].append(name)

    return clusters


def simulate_forest(num_nodes):
    forest = nx.DiGraph()

    for i in range(num_nodes):
        forest.add_node(i)

    root_nodes = range(num_nodes)

    while len(root_nodes) > 0:
        random.shuffle(root_nodes)

        node = root_nodes.pop()

        num_children = random.randint(0, len(root_nodes))

        children = random.sample(root_nodes, num_children)

        for child in children:
            forest.add_edge(node, child)

            root_nodes.remove(child)

        if random.uniform(0, 1) > 0.5:
            root_nodes.append(node)

    return forest
