# -*- coding: utf-8 -*-
"""
Created on Sat Mar 01 19:52:51 2014

@author: wu
"""
import igraph as ig

graph=ig.Graph.Watts_Strogatz(dim=1, size=8,nei=2, p=0.2, loops=False, multiple=False)
matrix=graph.get_adjacency()
print matrix
layout=graph.layout("kk")
ig.plot(graph,layout=layout)