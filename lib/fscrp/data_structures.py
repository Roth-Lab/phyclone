'''
Created on 16 Mar 2017

@author: Andrew Roth
'''
from collections import namedtuple

Particle = namedtuple('Particle', ['log_w', 'node', 'parent_particle'])

Node = namedtuple('Node', ['children', 'node_params', 'agg_params'])
