# -*- coding: utf-8 -*-
"""
Created on Sat Feb 01 16:49:55 2014

@author: wu
"""
import igraph as ig
import numpy as np
import test.Global.globalvars as vars
import os
from test.Global.HelpFunction import avg_wpl
import pandas as pd

"""
#simulate lattice
graph=ig.Graph.Watts_Strogatz(dim=1, size=8, nei=2, p=0.2, loops=False, multiple=False)
matrix=graph.get_adjacency()
print matrix
layout=graph.layout("kk")
ig.plot(graph,layout=layout)
"""





def simulateWeightedRandomGraph(p):
    """
    simulate weighted random graph
    the item in adjacent matrix is a number generated from geometric distribution(actually number-1),which its p=1-p in our case
    restrict the generated integer to be in the range of [0,MAX_WEIGHT]
    this can cause problem depending on how magnificant the probability p is
    Input: the probability for edge connection
    Output:graph from adjacent matrix
    """
    fail=1-p#the probability a connection is not generated
    #generate an adjacent matrix,2d array
    adjacent_matrix=np.zeros((vars.N,vars.N))
    #fill in the matrix symetrically
    for i in range(0,vars.N):
        for j in range(i+1,vars.N):
            rg_edge_weight = np.random.geometric(fail, size=1)
            rg_edge_weight=rg_edge_weight-1
            if rg_edge_weight>vars.MAX_WEIGHT:
                rg_edge_weight=vars.MAX_WEIGHT
            assert rg_edge_weight>=0 and rg_edge_weight<=vars.MAX_WEIGHT,"wrong edge weight is generated"
            adjacent_matrix[i,j]=rg_edge_weight
            adjacent_matrix[j,i]=rg_edge_weight
    adjacent_matrix=adjacent_matrix.tolist()
    graph=ig.Graph.Weighted_Adjacency(adjacent_matrix, mode=ig.ADJ_UNDIRECTED)
    #print adjacent_matrix
    return graph

simDir="C:\\Users\\wu\\Desktop\\WeightedRandomGraphSimulation"
p_range=np.linspace(0.05,0.95,19)
#p_range=[0.5]
generated_num=1000

"""
for p in p_range:
    p_table=np.zeros((generated_num,2))#record weighted local_transitivity and weighted average path length for every simulated graph
    for i in range(generated_num):
        graph=simulateWeightedRandomGraph(p)
        #save graph
        graphPath=simDir+"\\Graphs\\"+"p_"+str(p)
        if not os.path.isdir(graphPath):
            os.makedirs(graphPath)
        name=graphPath+"\\"+str(i)+".gml"
        graph.write_gml(name)
        #calculate graph measure
        local_transitivity=np.mean(graph.transitivity_local_undirected(vertices=graph.vs, mode="zero", weights="weight"))
        #g=graph.copy()
        #nomalize graph weight
        max=np.max(np.array(graph.es["weight"]))
        graph.es["weight"]=graph.es["weight"]/max
        apl=avg_wpl(graph)
        #save graph measure
        p_table[i]=(local_transitivity,apl)
    #convert p_table to dataframe
    p_table=pd.DataFrame(p_table,columns=["local_transitivity","apl"])
    #pickle dataframe into files
    tablePath=simDir+"\\Tables"
    if not os.path.isdir(tablePath):
        os.makedirs(tablePath)
    name=tablePath+"\\"+"p_"+str(p)+".csv"
    p_table.to_csv(name)
    print "mean: "
    print p_table.mean()
    print "standard deviation:"
    print p_table.std()
        


#layout=graph.layout("kk")
#ig.plot(graph,layout=layout)
"""






