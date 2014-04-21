# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:44:03 2014

@author: wu
"""

import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
import test.Global.globalvars as vars
import numpy as np
from test.I_O_put.Selector import *
from test.Global.HelpFunction import average_graph
import igraph as ig
from scipy.stats import ttest_ind

selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data

healthy_data,disease_data=data



#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,0.3331)
    centrality_list=graph.betweenness(directed=False,weights="weight")
    fig, ax = plt.subplots()
    pl.bar(np.arange(vars.N),centrality_list)

print "-----------------------------------------------------"
"""    
#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.3)
    
    
"""
#null hypothesis