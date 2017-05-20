'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from __future__ import division

from collections import namedtuple
from scipy.signal import fftconvolve

import copy
import numpy as np
import numba
import random

from fscrp.math_utils import log_sum_exp

Particle = namedtuple('Particle', ['log_w', 'node', 'parent_particle'])

MarginalParticle = namedtuple('MarginalParticle', ['log_w', 'parent_particle', 'nodes', 'node_idx', 'root_idxs'])

Node = namedtuple('Node', ['children', 'node_params', 'agg_params'])
#
# MarginalNode = namedtuple('MarginalNode', ['children', 'log_likelihoo', 'log_S'])


# class Particle(object):
#
#     def __init__(self, log_w, node, parent_particle):
#         self.log_w = log_w
#
#         self.node = node
#
#         self.parent_particle = parent_particle
#
#     def copy(self):
#         return Particle(self.log_w, self.node.copy(), self.parent_particle)
#
#
# class Node(object):
#
#     def __init__(self, children, node_params, agg_params):
#         self.children = children
#
#         self.node_params = node_params
#
#         self.agg_params = agg_params
#
#     def __key(self):
#         return (self.children, self.node_params, self.agg_params)
#
#     def __eq__(x, y):
#         return x.__key() == y.__key()
#
#     def __hash__(self):
#         return hash(self.__key())
#
#     def copy(self):
#         return Node(tuple(self.children), self.node_params, self.agg_params)


# class MarginalParticle(object):
# 
#     def __init__(self, block_idx, log_w, node, parent_particle, tree_log_p):
#         self.block_idx = block_idx
# 
#         self.log_w = log_w
# 
#         self.node = node
# 
#         self.parent_particle = parent_particle
# 
#         self.tree_log_p = tree_log_p
# 
#     def copy(self):
#         return copy.deepcopy(self)
#         return MarginalParticle(self.block_idx, self.log_w, self.node.copy(), self.parent_particle, self.tree_log_p)


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
        return copy.deepcopy(self)
#         new = MarginalNode(self.idx, [x.copy() for x in self.children], grid_size=self.grid_size)
#
#         new.log_likelihood = np.copy(self.log_likelihood)
#
#         new._update_log_R()
#
#         return new

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

#     def _compute_log_D_n(self, child_log_R, prev_log_D_n):
#         log_D_n = np.zeros(self.grid_size)
#
#         for i in range(self.grid_size):
#             temp = []
#
#             for j in range(self.grid_size):
#                 if j > i:
#                     break
#
#                 temp.append(child_log_R[j] + prev_log_D_n[i - j])
#
#             log_D_n[i] = log_sum_exp(temp)
#
#         return log_D_n

    def _update_log_R(self):
        self.log_R = self.log_likelihood + self.log_S

    def _update_log_S(self):
        self.log_S = np.zeros(self.grid_size)

        if len(self.children) > 0:
            log_D = self._compute_log_D()

            for i in range(self.grid_size[0]):
                #                 self.log_S[i, :] = _compute_log_S(log_D[i, :])

                self.log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])

#                 print self.log_S[i, :] - np.logaddexp.accumulate(log_D[i, :])
#                 print 'a', np.nansum(self.log_S[i, :] - np.logaddexp.accumulate(log_D[i, :]))


@numba.jit(nopython=True)
def _compute_log_D_n(child_log_R, prev_log_D_n):
    G = len(child_log_R)

    log_D_n = np.zeros(G)

    temp = np.zeros(G)

    for i in range(G):
        for j in range(i + 1):
            temp[j] = child_log_R[j] + prev_log_D_n[i - j]

        log_D_n[i] = log_sum_exp(temp[:i + 1])

    return log_D_n


@numba.jit(nopython=True)
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
