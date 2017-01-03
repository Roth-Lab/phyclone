# -*- coding: utf-8 -*-
'''
This file is part of PhyClone.

PhyClone is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PhyClone is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PhyClone.  If not, see
<http://www.gnu.org/licenses/>.

Created on 15-03-2013

@authors Alexandre Bouchard-Côté, Andrew Roth

Definition: A clade is a set of data_points, obtained as the set of data_points in the node n of a PhyClone tree plus all
the mutation in the children of n.
'''
from __future__ import division

from collections import defaultdict

import networkx as nx

def get_consensus_tree(graphs, threshold=0.5):
    clades_counter = clade_probabilities(graphs)
      
    consensus_clades = key_above_threshold(clades_counter, threshold)
    
    consensus_tree = consensus(consensus_clades)
    
    consensus_tree = relabel(consensus_tree)
    
    consensus_tree = clean_tree(consensus_tree)
    
    return consensus_tree  

def consensus(clades):
    '''
    Attempts to build a consensus tree from a set of clades. Returns a DiGraph where nodes are clades.
    '''
    result = nx.DiGraph()
    
    for clade in clades:
        candiate_supersets = clades.copy()
        
        parent_clade = find_smallest_superset(candiate_supersets, clade)
        
        if parent_clade != None:
            result.add_edge(parent_clade, clade)
        
        else:
            result.add_node(clade)
    
    return result

def relabel(graph):
    '''
    Relabels a consensus tree. Takes in a DiGraph of clades, return a new DiGraph where nodes are again set of mutation,
    but with a different interpretation. The tranformation used to change the nodes/sets is to start with the original
    and remove from each node the data_points that appear in children clades
    '''
    result = nx.DiGraph()
    
    for root in roots(graph):
        _relabel(root, result, graph)
    
    return result

def clean_tree(tree):
    new_tree = nx.DiGraph()
    
    node_ids = {}
    
    for i, node in enumerate(nx.dfs_preorder_nodes(tree)):
        node_ids[node] = "Node {0}".format(i + 1)
        
        new_tree.add_node(node_ids[node], data_points=sorted(node))
        
    for node in node_ids:
        for child in tree.successors(node):
            new_tree.add_edge(node_ids[node], node_ids[child])        

    return new_tree

def clade_probabilities(graphs):
    '''
    Reads a YAML file, loops over particles, normalize weights, then: return a dictionary where the entries are clades,
    and the values are posterior probabilities
    '''
    clades_counter = defaultdict(float)
    
    for graph, particle_probabilities in iter_graph_probabilities(graphs):
        data_points_map = get_data_points(graph)
                
        for clade in clades(graph, data_points_map):
            clades_counter[clade] += particle_probabilities
    
    return clades_counter

def key_above_threshold(counter, threshold):
    '''
    Only keeps the keys in a dict above or equal the threshold
    '''
    return set([key for key, value in counter.iteritems() if value > threshold])

def clades(graph, data_points):   
    result = set()

    for root in roots(graph):
        _clades(graph, root, result, data_points)
    
    return result

def _clades(graph, node, clades, data_points):
    current_clade = set()
    
    for mutation in data_points[node]: 
        current_clade.add(mutation)
    
    for _, children in graph.out_edges_iter(node):
        for mutation in _clades(graph, children, clades, data_points):
            current_clade.add(mutation)
    
    clades.add(frozenset(current_clade))
    
    return current_clade

def _relabel(node, transformed, original):
    result = set(node)
    
    for _, children in original.out_edges_iter(node):
        for mutation in children:
            result.remove(mutation)
    
    result = frozenset(result)
    
    transformed.add_node(result)
    
    for _, children in original.out_edges_iter(node):
        transformed.add_edge(result, _relabel(children, transformed, original))
    
    return result

def get_data_points(graph):
    result = {}

    for n, d in graph.nodes_iter(data=True):
        result[n] = set(mutation for mutation in d['data_points'])
    
    return result
    
def roots(graph): 
    return [n for n in graph.nodes_iter() if len(graph.in_edges(n)) == 0]

def increment(counter, key, increment):
    counter[key] = counter.get(key, 0) + increment
    
def find_smallest_superset(set_of_sets, query_set):
    # Remove the query set from set of candidate supersets if present
    set_of_sets.discard(query_set)
    
    # Intialisation
    smallest_superset_size = float('inf')

    smallest_superset = None
    
    # Loop through candidate supersets looking for the smallest one 
    for candidate_superset in set_of_sets:
        # Skip sets which aren't supersets of query
        if not candidate_superset.issuperset(query_set):
            continue
            
        candidate_superset_size = len(candidate_superset)

        if candidate_superset_size == smallest_superset_size: 
            raise Exception('Inconsistent set of clades')
        
        if candidate_superset_size < smallest_superset_size:
            smallest_superset_size = candidate_superset_size
            
            smallest_superset = candidate_superset
    
    return smallest_superset

def load_data_point_params(graphs):
    '''
    Args:
        particle_file : (path) Path of gzip compressed file with particles.
        
    Returns:
        data_point_params : (dict) A dictionary of lists with one entry per data point. Values are three tuples of 
        (particle_probability, agg_params, node_params).
    '''
    data_point_params = defaultdict(list)

    for graph, p in iter_graph_probabilities(graphs):
        for node in graph.nodes():
            for d in graph.node[node]['data_points']:
                data_point_params[d].append((p, graph.node[node]['agg_params'], graph.node[node]['node_params']))
    
    return data_point_params

def iter_graph_probabilities(graphs): 
    weights = []
    
    for i, graph in enumerate(graphs):
        if i % 100 == 0:
            print i
        
        weights.append(graph.graph['multiplicity'])
    
    probabilities = [x / sum(weights) for x in weights]

    for g, p in zip(graphs, probabilities):
        yield g, p