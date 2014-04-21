# -*- coding: utf-8 -*-
"""
Created on Sat Mar 01 22:33:24 2014

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
run=1
tsl=200#time series length


healthy_apl_list,disease_apl_list=[],[]
healthy_cc_list,disease_cc_list=[],[]

healthy_swIndex_list,disease_swIndex_list=[],[]
healthy_aplNormalized_list,healthy_ccNormalized_list=[],[]
disease_aplNormalized_list,disease_ccNormalized_list=[],[]

#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,"never used",isWeighted=True)
    W=get_edge_number(graph)
    print W
    apl=get_average_path_length(graph)
    cc=get_clustering_coefficient(graph)
    #calculate p
    p=2*W/((vars.N*(vars.N-1))+2*W)
    #simulate weighted random graph
    apl_,cc_=[],[]
    for i in range(run):
        graphlist=[]
        for j in range(tsl):
            graph=simulateWeightedRandomGraph(p)
            graphlist.append(graph)
        avg_graph=average_graph(graphlist,"never used",isWeighted=True)
        apl_.append(get_average_path_length(avg_graph))
        cc_.append(get_clustering_coefficient(avg_graph))
    #################################
    avg_apl=np.mean(np.asarray(apl_))
    avg_cc=np.mean(np.asarray(cc_))
    print type(avg_apl),type(avg_cc)
    aplNormalized=float(apl)/float(avg_apl)
    ccNormalized=float(cc)/float(avg_cc)
    swIndex=float(ccNormalized)/float(aplNormalized)
    healthy_swIndex_list.append(swIndex)
    healthy_aplNormalized_list.append(aplNormalized)
    healthy_ccNormalized_list.append(ccNormalized)

#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,"never used",isWeighted=True)
    W=get_edge_number(graph)
    apl=get_average_path_length(graph)
    cc=get_clustering_coefficient(graph)
    #calculate p
    p=2*W/((vars.N*(vars.N-1))+2*W)
    #simulate weighted random graph
    apl_,cc_=[],[]
    for i in range(run):
        graphlist=[]
        for j in range(tsl):
            graph=simulateWeightedRandomGraph(p)
            graphlist.append(graph)
        avg_graph=average_graph(graphlist,"never used",isWeighted=True)
        apl_.append(get_average_path_length(avg_graph))
        cc_.append(get_clustering_coefficient(avg_graph))
    #################################
    avg_apl=np.mean(np.asarray(apl_))
    avg_cc=np.mean(np.asarray(cc_))
    aplNormalized=float(apl)/float(avg_apl)
    ccNormalized=float(cc)/float(avg_cc)
    swIndex=float(ccNormalized)/float(aplNormalized)
    disease_swIndex_list.append(swIndex)
    disease_aplNormalized_list.append(aplNormalized)
    disease_ccNormalized_list.append(ccNormalized)



#plot
fig,ax=pl.subplots()
#left_limit=min([np.min(healthy_aplNormalized_list),np.min(disease_aplNormalized_list)])
upper_limit=max([np.max(healthy_aplNormalized_list),np.max(disease_aplNormalized_list)])
ax.set_xlim((0,upper_limit))
ax.set_ylim((0,upper_limit))
ax.scatter(healthy_aplNormalized_list,healthy_ccNormalized_list,color="r")
ax.scatter(disease_aplNormalized_list,disease_ccNormalized_list,color="b")
x=[0,upper_limit]
ax.plot(x,x)
ax.set_title("Small Worldness")
ax.set_xlabel("Normalized Weighted Average Path Length")
ax.set_ylabel("Normalized Weighted Clustering Coefficient")
pl.text(2,8,"Small World")
ax.grid(True)
ax.legend(["Healthy","Disease"])