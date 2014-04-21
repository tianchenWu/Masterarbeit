# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:13:34 2014

@author: wu
"""

from I_O_put.ReadDGS import ReadDGS
from Models.GraphModel import EEGGraph
import itertools
import numpy as np
import math
import Global.globalvars as vars
import gzip
import os

def get_plot_style(g):
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_label"] = g.vs["name"]
    if g.is_weighted(): visual_style["edge_width"] = [1 + 2 * int(weight) for weight in g.es["weight"]]
    visual_style["layout"] = "circle"
    #visual_style["bbox"] = (300, 300)
    visual_style["margin"] = 20
    return visual_style

"""
Method to process the Graph edge
Contiunous Weight
    0. weighted graph(original)
    1. weighted graph(thresholded)               using: thresholdize()
       result in weighted graph,edge has a fake-continuous weight

Discrete Weight

    2  unweighted graph(thresholded)             using: binarilize()
       prune the graph by cutting lowest PERCENTILE% weights
       result in non-weighted graph, edge is either existed or not
   
    3  weighted graph(discretized)               using: discretize()
       result in weighted Graph, edge is discrete value
"""

class WFilter:
    def __init__(self,dgs):
        self.filtered_graphlist=dgs.graphlist
        #graph pruning threshold
        self.threshold=None
    #get the threshold set
    def setThreshold(self):
        graph_weights_lists=[e.es["weight"] for e in self.filtered_graphlist]
        flattened_weights_list=list(itertools.chain(*graph_weights_lists))
        self.threshold=np.percentile(flattened_weights_list,vars.PERCENTILE)
        
    
    # filtering out the edge of which weight is strictly under the threshold
    def filterThreshold(self):
        for graph in self.filtered_graphlist:
            graph.delete_edges(graph.es.select(lambda x:float(x["weight"])<=self.threshold))
        return self.filtered_graphlist
    
    #convert the weighted graph to unweighted graph    
    def deWeight(self):
        for graph in self.filtered_graphlist:
            del graph.es["weight"]
            
    def thresholdize(self):
        self.setThreshold()
        self.filterThreshold()
        return self.filtered_graphlist
    
    #step 1: set up threshold
    #step 2: filter out under-weighted edges 
    #step 3: delete the weight attribute in Edge
    def binarilize(self):
        #print self.filtered_graphlist[0].es["weight"]
        self.setThreshold()
        self.filterThreshold()
        self.deWeight()
        #print "threshold: %s" %self.threshold
        return self.filtered_graphlist
    #discretize the weight of the graph
    def discretize(self):
        for graph in self.filtered_graphlist:
            for edge in graph.es:
                edge["weight"]=math.floor(edge["weight"]/vars.DISCRETIZATION_STEP)
        return self.filtered_graphlist
        pass


if __name__ == "__main__":
    """
    Sample Code
    """
    file_name=os.path.join(vars.rootDir,"Data","EEG-Graphen","NO2","Fedorov","GE","alpha_GE_prae_closed1.dgs.gz")
    f = gzip.open(file_name, 'rb')
    file_content = f.read()
    dgs=ReadDGS(file_content)
    g_before=dgs.graphlist[0]
    print "the number of edge before pruning: %d" %len(g_before.es)
    print "if weighted graph before pruning: %s" %g_before.is_weighted()
    import igraph as ig
    ig.plot(g_before,**(get_plot_style(g_before)))
     
    g_after=(WFilter(dgs).discretize())[0]
    
    print "the number of edge after pruning: %d" %len(g_after.es)
    print "if weighted graph after pruning : %s" %(g_after.is_weighted())
    if g_after.is_weighted(): print g_after.es["weight"]
    """
    g2=EEGGraph(WFilter(dgs).binarilize())
    g3=EEGGraph(WFilter(dgs).thresholdize())
    example_g2=g2.graphlist[0]
    example_g3=g3.graphlist[0]
    """
    ig.plot(g_after,**(get_plot_style(g_after)))
    