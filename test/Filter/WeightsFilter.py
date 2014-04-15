# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:13:34 2014

@author: wu
"""

from test.I_O_put.ReadDGS import ReadDGS
from test.Models.GraphModel import EEGGraph
import itertools
import numpy as np
import math
import test.Global.globalvars as vars
import gzip

"""
input: list of graphs
output: list of graphs
"""

##############################################################################
#Graph pruning threshold                                                     #
##############################################################################



'''
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
'''


##############################################################################
#take a list of temporal graphs as initialization parameter and do           #
#preprocessing on the list                                                   #
#threshold has to be set before filtering                                    #   
##############################################################################

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
        #print("threshold: %s" %(self.threshold))
    
    # filtering out the edge of which weight is strictly under the threshold
    def filterThreshold(self):
        for graph in self.filtered_graphlist:
            graph.delete_edges(graph.es.select(lambda x:float(x["weight"])<self.threshold))
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
        self.setThreshold()
        self.filterThreshold()
        self.deWeight()
        #print "after deWeight: %s" %self.filtered_graphlist[0].is_weighted()
        return self.filtered_graphlist
        
    
    #discretize the weight of the graph
    def discretize(self):
        for graph in self.filtered_graphlist:
            for edge in graph.es:
                edge["weight"]=math.floor(edge["weight"]/vars.DISCRETIZATION_STEP)
        #print self.filtered_graphlist[0].is_weighted()
        return self.filtered_graphlist


if __name__ == "__main__":
    file_name=vars.rootDir+"\\EEG-Graphen\\NO2\\Fedorov\\GE\\alpha_GE_prae_closed1.dgs.gz"
    f = gzip.open(file_name, 'rb')
    file_content = f.read()
    dgs=ReadDGS(file_content) 
    
    g1=EEGGraph(WFilter(dgs).discretize())
    example_g1=g1.graphlist[0]
    print "the number of edge before pruning: %d" %len(example_g1.es)
    print "if weighted graph : %s" %(example_g1.is_weighted())
    """
    g2=EEGGraph(WFilter(dgs).binarilize())
    #g=EEGGraph(WFilter(dgs).thresholdize())
    example_g2=g2.graphlist[0]
    print "the number of edge after pruning: %d" %len(example_g2.es)
    print "if weighted graph : %s" %(example_g2.is_weighted())
    """
    if example_g1.is_weighted():
        print " weights example : %s" %(example_g1.es["weight"])