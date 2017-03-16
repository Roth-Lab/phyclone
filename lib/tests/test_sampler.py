'''
Created on 2014-02-25

@author: andrew
'''
import numpy as np
import random
import unittest

from math import log
from pydp.densities import log_binomial_pdf
from pydp.rvs import binomial_rvs, poisson_rvs

from fscrp.sampler import Sampler, get_log_weights, get_nodes


class Test(unittest.TestCase):

    def testName(self):
        alpha = 1.0

        clusters = [[0.1, 0.5, 0.9], [0.2, 0.1, 0.02], [0.3, 0.6, 0.92], [1.0, 1.0, 1.0]]

        data = []

        i = 0

        for _ in range(10):
            for params in clusters:
                data_point = []

                for p in params:
                    n = poisson_rvs(1000)

                    x = binomial_rvs(n, p)

                    data_point.append(tuple([x, n]))

                data.append(tuple(data_point))

                i += 1

        print data

        num_particles = int(1e2)

        log_likelihood_func = lambda x, p: sum(
            [log_binomial_pdf(x_sample[0], x_sample[1], p_sample) for x_sample, p_sample in zip(x, p)])

        node_agg_func = lambda x: tuple(np.sum(x, axis=0))

        sampler = Sampler(
            alpha,
            data,
            num_particles,
            log_likelihood_func,
            param_proposal_func,
            node_agg_func,
            param_prior,
            num_implicit_particles=int(1e4),
            randomise_data=False
        )

        sampler.sample()

        best_log_weight = float('-inf')

        best_particle = None

        for particle in sampler.particle_multiplicities:
            log_w = sum(get_log_weights(particle)) + log(sampler.particle_multiplicities[particle])

            if log_w > best_log_weight:
                best_log_weight = log_w

                best_particle = particle

        for x in get_nodes(best_particle):
            print (x.node_params, x.agg_params)


def param_proposal_func(value, tree_params, precision=10):
    param = []

    log_q = 0

    for i, _ in enumerate(value):
        if len(tree_params) == 0:
            b = 1

        else:
            b = max(1 - sum([x[i] for x in tree_params]), 0)

        param.append(random.uniform(0, b))

        log_q += -log(b)

    return tuple(param), log_q


def param_prior(node, nodes):
    return len(node.node_params) * log(max(len(nodes), 1))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
