'''
Created on 9 Aug 2017

@author: Andrew Roth
'''
from __future__ import division

import random


def sample_sigma(tree, source=None):
    if source is None:
        sigma = []

        for node in tree.roots:
            sigma.append(sample_sigma(tree, source=node))

        outliers = list(tree.outliers)

        random.shuffle(outliers)

        sigma.append(outliers)

        return interleave_lists(sigma)

    child_sigma = []

    children = list(source.children)

    random.shuffle(children)

    for child in children:
        child_sigma.append(sample_sigma(tree, source=child))

    sigma = interleave_lists(child_sigma)

    source_sigma = list(tree.nodes[source.idx].data)

    random.shuffle(source_sigma)

    sigma.extend(source_sigma)

    return sigma


def interleave_lists(lists):
    result = []

    while len(lists) > 0:
        x = random.choice(lists)

        if len(x) == 0:
            lists.remove(x)

        else:
            result.append(x.pop(0))

    return result
