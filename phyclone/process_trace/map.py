import numpy as np
from phyclone.process_trace.utils import convert_rustworkx_to_networkx


def get_map_node_ccfs_and_clonal_prev_dicts(tree):
    graph = compute_map_tree_features(tree)

    ccf_dict = get_map_node_ccfs_dict(graph)
    clonal_prev_dict = get_map_node_clonal_prevs_dict(ccf_dict, graph)

    del ccf_dict["root"]
    del clonal_prev_dict["root"]
    return ccf_dict, clonal_prev_dict


def compute_map_tree_features(tree):
    graph = tree._graph.copy()
    graph = convert_rustworkx_to_networkx(graph)
    compute_max_likelihood(graph, "root")
    set_max_assignment(graph)
    return graph


def get_map_node_ccfs_dict(graph):
    ccf_dict = {}
    get_map_ccfs(graph, "root", ccf_dict)
    return ccf_dict


def get_map_node_clonal_prevs_dict(ccf_dict, graph):
    clonal_prev_dict = {}
    get_map_clonal_prev(graph, "root", ccf_dict, clonal_prev_dict)
    return clonal_prev_dict


def get_map_clonal_prev(tree, node, ccf_dict, result):
    clonal_prev = ccf_dict[node].copy()

    for child in tree.successors(node):
        clonal_prev -= ccf_dict[child]

        get_map_clonal_prev(tree, child, ccf_dict, result)

    result[node] = clonal_prev


def compute_max_likelihood(graph, node_id):
    children = list(graph.successors(node_id))

    for child_id in children:
        compute_max_likelihood(graph, child_id)

    node = graph.nodes[node_id]

    if len(children) == 0:
        node["log_S_max"] = np.zeros(node["log_p"].shape)

        node["log_R_max"] = node["log_p"]

    else:
        child_log_R = [graph.nodes[child_id]["log_R_max"] for child_id in children]

        node["log_D_choice"], node["log_S_choice"], node["log_S_max"] = compute_log_S(child_log_R)

        node["log_R_max"] = node["log_p"] + node["log_S_max"]


def compute_log_S(child_log_R_values):
    log_D_choice, log_D = compute_log_D(child_log_R_values)

    log_S = np.zeros(log_D.shape)

    log_S_choice = np.zeros(log_D.shape, dtype=int)

    grid_size = log_D.shape[1]

    num_dims = log_D.shape[0]

    log_S[:, 0] = log_D[:, 0]

    for i in range(num_dims):
        for j in range(1, grid_size):
            if log_D[i, j] > log_S[i, j - 1]:
                log_S[i, j] = log_D[i, j]

                log_S_choice[i, j] = j

            else:
                log_S[i, j] = log_S[i, j - 1]

                log_S_choice[i, j] = log_S_choice[i, j - 1]

    return log_D_choice, log_S_choice, log_S


def compute_log_D(child_log_R_values):
    log_D = np.zeros(child_log_R_values[0].shape)

    log_D_choice = []

    num_dims = log_D.shape[0]

    for child_log_R in child_log_R_values:
        child_choices = []

        for i in range(num_dims):
            choice, log_D[i, :] = _compute_log_D_n(child_log_R[i, :], log_D[i, :])

            child_choices.append(choice)

        log_D_choice.append(np.array(child_choices))

    return log_D_choice, log_D


def _compute_log_D_n(child_log_R, prev_log_D_n):
    grid_size = len(prev_log_D_n)

    choice = np.zeros(grid_size, dtype=int)

    result = np.ones(grid_size) * -np.inf

    for i in range(grid_size):
        for j in range(i + 1):
            val = child_log_R[j] + prev_log_D_n[i - j]

            if val >= result[i]:
                choice[i] = j

                result[i] = val

    return choice, result


def set_max_assignment(graph):
    num_dims, num_vals = graph.nodes["root"]["log_R"].shape

    graph.nodes["root"]["max_idx"] = np.ones(num_dims, dtype=int) * (num_vals - 1)

    _set_max_assignment(graph, np.ones(num_dims, dtype=int) * (num_vals - 1), "root")


def _set_max_assignment(graph, idxs, node):
    num_dims = len(idxs)

    children = list(graph.successors(node))

    if len(children) == 0:
        return

    child_total_idx = [graph.nodes[node]["log_S_choice"][d, idxs[d]] for d in range(num_dims)]

    for i in range(len(children) - 1, -1, -1):
        child = children[i]

        graph.nodes[child]["max_idx"] = np.zeros(num_dims, dtype=int)

        for d in range(num_dims):
            graph.nodes[child]["max_idx"][d] = graph.nodes[node]["log_D_choice"][i][d, child_total_idx[d]]

            child_total_idx[d] -= graph.nodes[child]["max_idx"][d]

        _set_max_assignment(graph, graph.nodes[child]["max_idx"], children[i])


def get_map_ccfs(graph, node, result):
    num_dims = graph.nodes[node]["log_R"].shape[1]

    result[node] = np.array([x / (num_dims - 1) for x in graph.nodes[node]["max_idx"]])

    for child in graph.successors(node):
        get_map_ccfs(graph, child, result)
