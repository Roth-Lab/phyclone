import networkx as nx
import rustworkx as rx


def convert_rustworkx_to_networkx(graph):
    """Convert a rustworkx PyGraph or PyDiGraph to a networkx graph."""
    edge_list = [(
        graph[x[0]].node_id, graph[x[1]].node_id,
        {'weight': x[2]}) for x in graph.weighted_edge_list()]

    if isinstance(graph, rx.PyGraph):
        # if graph.multigraph:
        #     return nx.MultiGraph(edge_list)
        # else:
        #     return nx.Graph(edge_list)
        return nx.Graph(edge_list)
    else:
        # if graph.multigraph:
        #     return nx.MultiDiGraph(edge_list)
        # else:
        #     return nx.DiGraph(edge_list)
        nx_graph = nx.DiGraph(edge_list)
        for node in graph.nodes():
            node_id = node.node_id
            nx_node = nx_graph.nodes[node_id]
            nx_node.update(node.to_dict())

        return nx_graph


def print_string_to_file(str_to_write, filename):
    with open(filename, "w") as f:
        print(str_to_write, file=f)
