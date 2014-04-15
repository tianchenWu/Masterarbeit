# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 18:27:34 2014

@author: wu
"""

import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import test.Global.globalvars as vars
import numpy as np
from test.I_O_put.Selector import *
from test.Global.HelpFunction import average_graph
import igraph as ig




selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data

healthy_data,disease_data=data

healthy_edge_list,disease_edge_list=[],[]
healthy_apl_list,disease_apl_list=[],[]
healthy_cc_list,disease_cc_list=[],[]

healthy_swIndex_list,disease_swIndex_list=[],[]

#########################################################################
#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,0.3,isWeighted=False)
    expected_edge=int(get_edge_number(graph))
    ##########################################Random Graph
    run=100
    apl_,cc_=[],[]
    
    for i in range(run):
        graph=ig.Graph.Erdos_Renyi(n=vars.N,m=expected_edge, directed=False, loops=False)
        apl_.append(get_average_path_length(graph))
        cc_.append(get_clustering_coefficient(graph))
    
    avg_apl,std_apl=np.mean(np.asarray(apl_)),np.std(np.asarray(apl_))
    avg_cc,std_cc=np.mean(np.asarray(cc_)),np.std(np.asarray(cc_))
    print "random graph std_apl: %s" %std_apl
    print "random graph std_cc:  %s" %std_cc
    #################################################
    aplNormalized=float(get_average_path_length(graph))/float(avg_apl)
    ccNormalized=float(get_clustering_coefficient(graph))/float(avg_cc)
    swIndex=float(ccNormalized)/float(aplNormalized)
    healthy_swIndex_list.append(swIndex)
#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.3,isWeighted=False)
    expected_edge=int(get_edge_number(graph))
    ##########################################Random Graph
    run=100
    apl_,cc_=[],[]
    
    for i in range(run):
        graph=ig.Graph.Erdos_Renyi(n=vars.N,m=expected_edge, directed=False, loops=False)
        apl_.append(get_average_path_length(graph))
        cc_.append(get_clustering_coefficient(graph))
    
    avg_apl,std_apl=np.mean(np.asarray(apl_)),np.std(np.asarray(apl_))
    avg_cc,std_cc=np.mean(np.asarray(cc_)),np.std(np.asarray(cc_))
    print "random graph std_apl: %s" %std_apl
    print "random graph std_cc:  %s" %std_cc
    #################################################
    aplNormalized=float(get_average_path_length(graph))/float(avg_apl)
    ccNormalized=float(get_clustering_coefficient(graph))/float(avg_cc)
    swIndex=float(ccNormalized)/float(aplNormalized)
    disease_swIndex_list.append(swIndex)
    
print healthy_swIndex_list,disease_swIndex_list
print np.mean(np.asarray(healthy_swIndex_list)),np.mean(np.asarray(disease_swIndex_list))
print np.std(np.asarray(healthy_swIndex_list)),np.std(np.asarray(disease_swIndex_list))
"""
#plot
fig,ax=pl.subplots()
index=np.linspace(0,1.0,num=20)
ax.plot(index,healthy_swIndex_list,color="r")
ax.plot(index,disease_swIndex_list,color="b")
ax.set_title("Relationship between Small World Index and Time Series Threshold")
ax.set_xlabel("Threshold")
ax.set_ylabel("Small World Index")
ax.grid(True)
ax.legend(["Healthy","Disease"])
"""