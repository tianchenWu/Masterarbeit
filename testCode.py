# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:51:31 2014

@author: wu
"""

import igraph as ig

def weighted_clustering_coefficient(g,vertice,weights="weight"):
    """
    vertex transitivities of the graph. Vertices with less than two neighbors require special treatment, 
    they will considered as having zero transitivity
    Calculation is based on the formula in paper
    A Weighted small world network measure for assessing functional connectivity
    formula (8)
    """
    k_i=g.degree(vertice)
    print "k_i= %s" %k_i
    cc=0
    if k_i<=2:
        return cc
            
    neighbors=g.neighbors(vertice)
    print neighbors 
    #calculate for numerator 3*(sum w_ij,w_jh,w_hi),called part I,noted P1
    p1=sum([g[vertice,j]*g[vertice,h]*g[j,h] for j in neighbors for h in neighbors])  
    #calculate for the left part of denomerator (sum w_ih w_hj)
    p2=sum([g[vertice,h]*g[h,j] for j in neighbors for h in neighbors])
    cc=float(3*p1)/float(2*p2+p1)
    return cc

def avg_wcc(g,weights="weight"):
    sum_wcc=sum([weighted_clustering_coefficient(g,v,weights=weights) for v in g.vs])
    return float(sum_wcc)/float(len(g.vs))
    
    
    
    
    
g =ig.Graph()
g.add_vertices(4)
g.vs["label"]=["0","1","2","3",]
g.add_edges([(0,1),(1,2),(0,3),(2,3),(2,0),(1,3)])
g.es["weight"]=[0.1,0.5,1,0.5,0.2,0.5]
#g.delete_edges((1,2),(3,4))

print g.vs[1]

for e in g.es:
    print "edge weight: %s" %e["weight"]
    
#count=g.transitivity_local_undirected(vertices=g.vs[0], mode="zero", weights="weight")
count1=g.transitivity_avglocal_undirected(mode="zero", weights="weight")
print "weighted clustering coefficient by Barrat: %s" %count1


count2=avg_wcc(g)
print "weighted clustering coefficient calculated by Bolanos: %s" %count2
layout=g.layout("kk")
ig.plot(g,layout=layout)