# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:34:27 2014

@author: wu
"""

import numpy as np
import test.Global.globalvars as vars
from test.I_O_put.Selector import Selector
import math
from test.StatsAnalysis.GraphMeasuresTable import *
import matplotlib.pylab as pl


range_apl=1#within +/- of the imaginary apl
range_cc=0.6#within +/- of the imaginary cc

def RGFilter(EEGGraph,method=None):
    """
    Random Graph Model Filter
    EEG time series is modeled by point estimated single parameter p in random graph model, for each random graph(weighted
    or no weighted) with certain parameter p, there should be with high probability also other properties available. By
    checking those properties corresponding to given p, we can filter out those graphs which are very likely not generated
    by this parameter under random graph model.
    
    The properties we check against are average path length and clustering coefficient

    Unlike GS smoothing filter, RGFilter causes the shortening of the time series.    
    
    Input: 1 EEGGraph  2 Filter method and parameters
    Output: GraphModel #filtered p in 2d array ,2d-array time series graph measure in the shape of (n_samples,n_measures)
    """    
    RGraphModel_filtered=[]
    graphlist=EEGGraph.graphlist# iGraph Object list
    p_list=EEGGraph.measures#2d array
    assert len(p_list[0])==1,"data is not random graph modeled, please use other filter"
    for graph,p in zip(graphlist,p_list):
        measure_table=estGraphMeasureTable(graph)
        #extract apl and cc in reality
        re_apl=measure_table["apl"]
        re_cc=measure_table["transitivity_global"]
        #print "re_apl:{0:.3f}".format(re_apl[0])
        #print "re_cc:{0:.3f}".format(re_cc[0])
        im_apl,im_cc=constructImagineProperties(p)

        #if real apl is in the imaginary range
        constraint1= True if (re_apl>(im_apl+(-1)*range_apl)) and (re_apl<(im_apl+range_apl)) else False
        constraint2= True if ((re_cc>(im_cc+(-1)*range_cc)) and (re_cc<(im_cc+range_cc))) else False
        #only when both contraints are fullfilled, this time stamp will be included
        if constraint1 and constraint2:
            RGraphModel_filtered.append(p)
        else:
            pass
    return RGraphModel_filtered# list, 1d

def constructImagineProperties(p):
    """
    Input: generative graph parameters, p in case of RG
    Output: should-be properties  average path length and global transitivity in this case
    """
    #compute imagining apl and cc
    #if p<= 1/N , that means for random graph that the graph is not connected,so that
    #the average path length might be infinity
    #to be consistent with the calculation of average path length of igraph
    # a path length equal to the number of vertices is used when network is not connected
    if(math.log(p*vars.N)<=0):
        im_apl=vars.N
    else:  
        im_apl=math.log(vars.N)/math.log(p*vars.N)
    if(im_apl>vars.N):
        im_apl=vars.N
    assert im_apl<vars.N+1,"apl has been calculated wrongly"
    im_cc=p[0]
    #print "im_apl:{0:.3f}".format(im_apl)
    #print "im_cc:{0:.3f}".format(im_cc)
    return im_apl,im_cc

def RGFilterMeasure(EEGGraph):
    """
    some information about random graph filter and plot them
    """
    original_graphlist=EEGGraph.measures
    filtered_RGraphModel=RGFilter(EEGGraph)
    original_len=len(original_graphlist)
    print "original length: {0:.3f}".format(original_len)
    filtered_len=len(filtered_RGraphModel)
    print "after filter length: {0:.3f}".format(filtered_len)
    #percentage to be cutted
    cutted=1-float(filtered_len)/float(original_len)
    
    original_m=NA(EEGGraph).extractInfo()[["apl","transitivity_local"]]#original apl,transitivity_global  dataFrame
    print "type of original_apl: %s" %original_m
    print original_m
    im_apl_list=[]
    im_cc_list=[]
    im_apl_lb=[]# lower boundary list
    im_apl_ub=[]#upper boundary list
    im_cc_lb=[]
    im_cc_ub=[]
    #reconstruct imagine properties
    for p in original_graphlist:
        im_apl,im_cc=constructImagineProperties(p)
        im_apl_list.append(im_apl)
        im_cc_list.append(im_cc)
        
        im_apl_lb.append(im_apl-range_apl)
        im_apl_ub.append(im_apl+range_apl)
        im_cc_lb.append(im_cc-range_cc)
        im_cc_ub.append(im_cc+range_cc)
    
    re_apl=original_m["apl"]#series
    re_cc=original_m["transitivity_local"]#series
    #plot real properties against imaginary properties
    pl.figure()
    x=np.arange(original_len)
    pl.plot(x,im_apl_list,"b",x,im_apl_lb,"c--+",x,im_apl_ub,"c--+")
    pl.plot(x,re_apl,"k")
    pl.xlabel("Time Series")
    pl.ylabel("Number of Average Path Length")
    pl.legend(("theoretical apl","boundary of imaginary apl","boundary of imaginary apl","real apl"))
    #pl.legend(("imaginary apl","real apl"))
    pl.title("Average Path Length Filter")
    
    pl.figure()
    pl.plot(x,im_cc_list,"b")
    pl.plot(x,re_cc,"k")
    pl.xlabel("Time Series")
    pl.ylabel("Clustering Coefficient")
    pl.legend(("theoretical clustering coefficient","real clustering coeffcient"))  
    
    #graph measure(p) before and after filter    
    pl.figure()
    pl.title("p before and after filter")
    p_list=[l[0] for l in EEGGraph.measures]
    print p_list
    pl.plot(np.arange(len(p_list)),p_list,"k")
    pl.plot(np.arange(filtered_len),filtered_RGraphModel,"r")
    pl.xlabel("Time Series")
    pl.ylabel("p parameter")
    pl.legend(("before filter","after filter"))

        
    
if __name__=="__main__":
   selector=Selector("2",weight_filter_method="binarilize")
   data=selector.data
   for file_name,g in data:
       RGFilterMeasure(g)
       #filtered=RGFilter(g)



    
    
    