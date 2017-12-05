'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import namedtuple
from scipy.signal import fftconvolve

import numpy as np
import numba

from fscrp.math_utils import log_sum_exp

MarginalParticle = namedtuple('MarginalParticle', ['log_w', 'parent_particle', 'nodes', 'node_idx', 'root_idxs'])


class MarginalNode(object):

    def __init__(self, idx, grid_size, children=None):
        self.idx = idx

        self.grid_size = grid_size

        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.log_likelihood = np.ones(grid_size) * -np.log(grid_size[1])

        self.log_R = np.zeros(grid_size)

        self.update()

    @property
    def children(self):
        return self._children.values()

    @property
    def log_p(self):
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += log_sum_exp(self.log_R[i, :])

        return log_p

    @property
    def log_p_one(self):
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += self.log_R[i, -1]

        return log_p

    def add_child_node(self, node):
        assert node.idx not in self.children

        self._children[node.idx] = node

        self.update()

    def remove_child_node(self, node):
        del self._children[node.idx]

        self.update()

    def update_children(self, children):
        self._children = {}

        if children is not None:
            for child in children:
                self._children[child.idx] = child

        self.log_R = np.zeros(self.grid_size)

        self.update()

    def add_data_point(self, data_point):
        '''
        Add a data point to the collection at this node.
        '''
        self.log_likelihood += data_point

        self._update_log_R()

    def remove_data_point(self, data_point):
        '''
        Add a data point to the collection at this node.
        '''
        self.log_likelihood -= data_point

        self._update_log_R()

    def copy(self):
        #TODO: Replace call to __init__ with __new__ and skip call to update()
        new_children = [x.copy() for x in self.children]

        new = MarginalNode(self.idx, self.grid_size, children=new_children)

        new.log_likelihood = np.copy(self.log_likelihood)

        new.log_R = np.copy(self.log_R)

        new.log_S = np.copy(self.log_S)

        return new

    def update(self):
        self._update_log_S()

        self._update_log_R()

    def _compute_log_D(self):
        for child_id, child in enumerate(self.children):
            if child_id == 0:
                log_D = child.log_R.copy()

            else:
                for i in range(self.grid_size[0]):
                    log_D[i, :] = _compute_log_D_n(child.log_R[i, :], log_D[i, :])

        return log_D

    def _update_log_R(self):
        self.log_R = self.log_likelihood + self.log_S

    def _update_log_S(self):
        self.log_S = np.zeros(self.grid_size)

        if len(self.children) > 0:
            log_D = self._compute_log_D()

            for i in range(self.grid_size[0]):
                self.log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])


def _compute_log_D_n(child_log_R, prev_log_D_n):
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = fftconvolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max


@numba.jit(nopython=True, cache=True)
def _compute_log_S(log_D):
    G = len(log_D)

    log_S = np.zeros(G)

    temp = np.zeros(2)

    log_S[0] = -np.inf

    for i in range(1, G):
        temp[0] = log_D[i]

        temp[1] = log_S[i - 1]

        log_S[i] = log_sum_exp(temp)

    return log_S
