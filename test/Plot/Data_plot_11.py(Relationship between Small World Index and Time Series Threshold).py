# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 18:29:32 2014

@author: wu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:25:14 2014

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

#time series threshold  THRESHOLD=threshold*length
threshold_list=np.linspace(0,1.0,num=20)
for t in threshold_list:
    #healthy group
    for file_name,eeg in healthy_data:
        graph=average_graph(eeg.graphlist,t,isWeighted=False)
        edgeNumber=get_edge_number(graph)
        healthy_edge_list.append(edgeNumber)
        healthy_apl_list.append(get_average_path_length(graph))
        healthy_cc_list.append(get_clustering_coefficient(graph))
    #eye disease group
    for file_name,eeg in disease_data:
        graph=average_graph(eeg.graphlist,t,isWeighted=False)
        edgeNumber=get_edge_number(graph)
        disease_edge_list.append(edgeNumber)
        disease_apl_list.append(get_average_path_length(graph))
        disease_cc_list.append(get_clustering_coefficient(graph))
        
    #expected edge
    healthy_edge_list.extend(disease_edge_list)
    expected_edge=int(np.floor(np.mean(np.asarray(healthy_edge_list))))
    healthy_aplMean,disease_aplMean=np.mean(np.asarray(healthy_apl_list)),np.mean(np.asarray(disease_apl_list))
    healthy_ccMean,disease_ccMean=np.mean(np.asarray(healthy_cc_list)),np.mean(np.asarray(disease_cc_list))
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
    
    healthy_aplNormalized=float(healthy_aplMean)/float(avg_apl)
    print healthy_aplNormalized
    disease_aplNormalized=float(disease_aplMean)/float(avg_apl)
    print disease_aplNormalized
    healthy_ccNormalized=float(healthy_ccMean)/float(avg_cc)
    print healthy_ccNormalized
    disease_ccNormalized=float(disease_ccMean)/float(avg_cc)
    print disease_ccNormalized

    healthy_swIndex=float(healthy_ccNormalized)/float(healthy_aplNormalized)
    disease_swIndex=float(disease_ccNormalized)/float(disease_aplNormalized)
    healthy_swIndex_list.append(healthy_swIndex)
    disease_swIndex_list.append(disease_swIndex)

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
    