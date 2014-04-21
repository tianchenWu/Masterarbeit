# -*- coding: utf-8 -*-
"""
Created on Sat Mar 01 20:12:08 2014

@author: wu
"""

import numpy as np
import test.Global.globalvars as vars
from test.I_O_put.Selector import Selector
import math
from test.StatsAnalysis.GraphMeasuresTable import *
import matplotlib.pylab as pl





def RGFilterMeasure(EEGGraph):
    """
    some information about random graph filter and plot them
    """
    original_graphlist=EEGGraph.measures
    original_len=len(original_graphlist)
    
    original_m=NA(EEGGraph).extractInfo()[["apl","transitivity_local"]]#original apl,transitivity_global  dataFrame
    print "type of original_apl: %s" %original_m
    print original_m

    rg_apl_list=[]
    rg_cc_list=[]



    for p in original_graphlist:
        ##########################################Simulate RG
        run=10000
        apl_,cc_=[],[]
        expected_edge=np.floor(p*(vars.N)*(vars.N-1)*0.5)
        for i in range(run):
            graph=ig.Graph.Erdos_Renyi(n=vars.N,m=expected_edge, directed=False, loops=False)
            apl_.append(get_average_path_length(graph))
            cc_.append(get_clustering_coefficient(graph))
        
        avg_apl,std_apl=np.mean(np.asarray(apl_)),np.std(np.asarray(apl_))
        avg_cc,std_cc=np.mean(np.asarray(cc_)),np.std(np.asarray(cc_))
        
        rg_apl_list.append(avg_apl)
        rg_cc_list.append(avg_cc)
        ##########################################
        
    re_apl=original_m["apl"]#series
    re_cc=original_m["transitivity_local"]#series
    
    swi_list=[]#small world index
    for i in np.arange(original_len):
        aplNormalized=float(re_apl[i])/float(rg_apl_list[i])
        ccNormalized=float(re_cc[i])/float(rg_cc_list[i])

        swIndex=float(ccNormalized)/float(aplNormalized)
        swi_list.append(swIndex)
    
    assert len(re_apl)==len(re_cc) and len(re_cc)==len(rg_apl_list) and original_len==len(rg_apl_list),"Length is not correct"

    #plot real properties against imaginary properties
    fig,(ax1,ax2)=pl.subplots(2)
    x=np.arange(original_len)
    ax1.plot(x,rg_apl_list,"b")
    ax1.plot(x,re_apl,"k")
    ax1.set_xlabel("Time Series")
    ax1.set_ylabel("Average Path Length")
    ax1.legend(("Random Graph apl","Dataset apl"))
    ax1.grid(True)
    
    ax2.plot(x,rg_cc_list,"b")
    ax2.plot(x,re_cc,"k")
    ax2.set_xlabel("Time Series")
    ax2.set_ylabel("Clustering Coefficient")
    ax2.legend(("Random Graph clustering coefficient","Dataset clustering coeffcient"))  
    ax2.grid(True)
    #small world index   
    fig,ax3=pl.subplots()
    ax3.set_title("Small World Index along Time Series")
    ax3.plot(x,swi_list,"k")
    ax3.set_xlabel("Time Series")
    ax3.set_ylabel("Small World Index")
    ax3.axhline(y=1,color="r")
    ax3.grid(True)
        
    
if __name__=="__main__":
   selector=Selector("2",weight_filter_method="binarilize")
   data=selector.data
   for file_name,g in data:
       RGFilterMeasure(g)
