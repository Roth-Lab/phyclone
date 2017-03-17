'''
Created on 17 Mar 2017

@author: Andrew Roth
'''
import random


def get_labels(graph):
    labels = {}

    for node in graph.nodes():
        for data_point in graph.node[node]['data_points']:
            labels[data_point] = node

    return [labels[x] for x in sorted(labels)]


def get_roots(graph):
    return [x for x in graph.nodes_iter() if graph.in_degree(x) == 0]


def sample_sigma(graph, source=None):
    if source is None:
        roots = get_roots(graph)

        sigma = []

        for node in roots:
            sigma.append(sample_sigma(graph, source=node))

        return interleave_lists(sigma)

    child_sigma = []

    for child in graph.successors(source):
        child_sigma.append(sample_sigma(graph, source=child))

    sigma = interleave_lists(child_sigma)

    source_sigma = list(graph.node[source]['data_points'])

    random.shuffle(source_sigma)

    sigma.extend(source_sigma)

    return sigma


def interleave_lists(lists):
    result = []

    while len(lists) > 0:
        x = random.choice(lists)

        result.append(x.pop(0))

        if len(x) == 0:
            lists.remove(x)

    return result
