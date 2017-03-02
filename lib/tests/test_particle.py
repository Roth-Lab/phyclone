'''
Created on 2014-02-22

@author: andrew
'''
import unittest

import random

from fscrp.forest import Forest
from fscrp.particle import Particle


class Test(unittest.TestCase):

    def testName(self):
        data = range(10)

        forest = Forest()

        particle = Particle(data, forest, ll, h, sum)

        particle.extend_particle_extension(1, 1, 0)

        particle.extend_particle_extension(1, 2, 0)


def ll(x, p):
    return x ** p


def h(x, vals):
    return random.normalvariate(0, 1), 1e-3

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
