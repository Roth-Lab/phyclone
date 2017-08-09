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

    def __init__(self, idx, children, grid_size):
        self.idx = idx

        self.grid_size = grid_size

        self.children = tuple(children)

        self.log_likelihood = np.ones(grid_size) * -np.log(grid_size[1])

        self.log_R = np.zeros(grid_size)

        self.update()

    def __key(self):
        return (self.idx, self.children, tuple(self.log_likelihood.flatten()))

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def log_p(self):
        log_p = 0

        for i in range(self.grid_size[0]):
            log_p += log_sum_exp(self.log_R[i, :])

        return log_p

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
        new = MarginalNode(self.idx, [x.copy() for x in self.children], grid_size=self.grid_size)

        new.log_likelihood = np.copy(self.log_likelihood)

        new.log_R = np.copy(self.log_R)

        new.log_S = np.copy(self.log_S.copy())

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
