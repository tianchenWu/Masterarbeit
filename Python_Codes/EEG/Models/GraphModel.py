# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:03:34 2014

@author: wu
"""

import igraph as ig
import numpy as np
import gzip
import Global.globalvars as vars
from I_O_put.ReadDGS import ReadDGS
import math
from Global.HelpFunction import avg_wpl,avg_wcc
#from Selector import Selector


"""
Input: 1)temporal EEG Graphs of one patient 2)graph measures
        
Output: a nxm dimension ndarray measures on the given temporal graph
"""
class EEGGraph:

    
    def __init__(self,graphlist):
        #global graph_measurenames
        self.graphlist=graphlist
        #nxm ndarray, n is the length of time series, m is the number of measures
        self.n=len(graphlist)
        self.m=len(vars.graph_measurenames)
        self.measures=np.zeros((self.n,self.m))     
        
        self.vertex_name=self.graphlist[0].vs["name"]
        self.ds=[]     #degree sequence
           
        self.getMeasures()
    
    def degreeDistribution(self):
        self.ds=[sum(t) for t in zip(*self.ds)]
        #print (self.ds)
    
    def calculateDS(self):
        """
        Calculate the degree sequence for each graph
        Output: list of array  [[degree sequence for graph 1],[degree sequence for graph 2]....] 
        """
        for graph in self.graphlist:
            degree_sequence=np.asarray(graph.degree(graph.vs))
            self.ds.append(degree_sequence)            
    
    def getMeasures(self):
        
        for (c1,measure) in enumerate(vars.graph_measurenames):         
            if measure=="density":
                for (c2,graph) in enumerate(self.graphlist):
                    self.measures[c2,c1]=graph.density(loops=False)
            if measure=="clustering coefficient":
                for (c2,graph) in enumerate(self.graphlist):
                    if not graph.is_weighted():
                        self.measures[c2,c1]=graph.transitivity_avglocal_undirected(mode="zero")
                    if graph.is_weighted():
                        self.measures[c2,c1]=avg_wcc(graph,weights="weight")
            if measure=="average path length":
                for (c2,graph) in enumerate(self.graphlist):
                    if not graph.is_weighted():
                        self.measures[c2,c1]=graph.average_path_length(directed=False, unconn=False)
                    if graph.is_weighted():
                        g=graph.copy()
                        #nomalize graph weight
                        max=np.max(np.array(graph.es["weight"]))
                        g.es["weight"]=g.es["weight"]/max
                        #print g.es["weight"]
                        #calculated weigted avg path length on normalized graph
                        self.measures[c2,c1]=avg_wpl(g)
            if measure=="RG_p":
                for (c2,graph) in enumerate(self.graphlist):
                    #p can be estimated by #edges/[n(n-1)]
                    self.measures[c2,c1]=float(len(graph.es))/(vars.N*(vars.N-1))
            if measure=="WRG_p":
                assert self.graphlist[0].is_weighted(),"The Graph is not weighted, can not take weight attribute of the edge"
                for (c2,graph) in enumerate(self.graphlist):
                    #p can be estimated by 2W/[N*(N-1)+2*W]
                    #W is the sum of edge weights in the real network
                    W=sum(graph.es["weight"])
                    self.measures[c2,c1]=2*W/((vars.N*(vars.N-1))+2*W)
            if measure=="TWS":
                self.measures=np.zeros((self.n,2))
                self.calculateDS()
                for (c2,graph) in enumerate(self.graphlist):
                    global_transitivity=graph.transitivity_undirected(mode="nan")
                    #print self.ds[c2]
                    K=np.floor(self.ds[c2].mean())
                    #assert K>1, "graph is too sparse for the method, K is equal or below 1"
                    #if K is too small, consider the graph as noise
                    if K<=1:
                        print "small k"
                        K=2
                    if K>2:
                        C_0=0.75*float((K-2))/float((K-1))
                        #print "K: %s" %K
                        # c(0)=3(K-2)/4(K-1)  global_transitivity=. C(0)(1-p)^3
                        #print "C_0: %s" %C_0
                        p=1-float(math.pow(float(global_transitivity)/float(C_0),float(1)/float(3)))
                        if p<0:
                            p=0
                        assert p>=0, "p shouldn't be smaller than 0"
                    #K=2 estimate p using (measured global transitivity/imagined global transitivity from random graph)
                    if K==2:
                        imagined_global_transitivity=float(len(graph.es))/(vars.N*(vars.N-1))
                        p=float(global_transitivity)/float(imagined_global_transitivity)
                    #print "p: %s" %p
                    self.measures[c2]=np.asarray([K,p])
                    
            if measure=="degree sequence":
                #reset the 2-dimension of g.measures
                self.measures=np.zeros((self.n,vars.N))                
                #print "shape of degree sequence %s" %(self.measures.shape)
                for (c2,graph) in enumerate(self.graphlist):
                    self.measures[c2]=np.asarray(graph.degree(graph.vs))
            if measure=="gcor":
                num_elements_half_adjmatrix=(np.asarray((self.graphlist[0].get_adjacency(type=0).data)).flatten()).shape[0]
                self.measures=np.zeros((self.n,num_elements_half_adjmatrix))
                for (c2,graph) in enumerate(self.graphlist):
                    self.measures[c2]=np.asarray(graph.get_adjacency(type=0).data).flatten()
                    
                    
                    
                    
                    
                    
                    
                    
        