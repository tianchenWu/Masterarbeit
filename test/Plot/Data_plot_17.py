# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 17:50:15 2014

@author: wu
"""

import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import test.Global.globalvars as vars
import numpy as np
from test.I_O_put.Selector import *
from test.Global.HelpFunction import average_graph,avg_wcc
import igraph as ig
from scipy.stats import ttest_ind
from test.StatsAnalysis.NetworkSimulation import simulateWeightedRandomGraph


"""
hub is calculated by maximal weighted betweenness centrality of all vertices in a graph
"""
selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data

healthy_data,disease_data=data

healthy_edge_list,disease_edge_list=[],[]
healthy_apl_list,disease_apl_list=[],[]
healthy_cc_list,disease_cc_list=[],[]


#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,"never used",isWeighted=True)    
    edgeNumber=get_edge_number(graph)
    healthy_edge_list.append(edgeNumber)
    healthy_apl_list.append(get_average_path_length(graph))
    healthy_cc_list.append(avg_wcc(graph))

print "-----------------------------------------------------"

#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,"never used",isWeighted=True)
    edgeNumber=get_edge_number(graph)
    disease_edge_list.append(edgeNumber)
    disease_apl_list.append(get_average_path_length(graph))
    disease_cc_list.append(avg_wcc(graph))


#average over the number of patients in the group, 15
numPatient=15
healthy_edgeNumberMean,disease_edgeNumberMean=np.mean(np.asarray(healthy_edge_list)),np.mean(np.asarray(disease_edge_list))
healthy_aplMean,disease_aplMean=np.mean(np.asarray(healthy_apl_list)),np.mean(np.asarray(disease_apl_list))
healthy_ccMean,disease_ccMean=np.mean(np.asarray(healthy_cc_list)),np.mean(np.asarray(disease_cc_list))

healthy_edgeNumberStd,disease_edgeNumberStd=np.std(np.asarray(healthy_edge_list)),np.std(np.asarray(disease_edge_list))
healthy_aplStd,disease_aplStd=np.std(np.asarray(healthy_apl_list)),np.std(np.asarray(disease_apl_list))
healthy_ccStd,disease_ccStd=np.std(np.asarray(healthy_cc_list)),np.std(np.asarray(disease_cc_list))


#EX1
print healthy_edgeNumberMean,disease_edgeNumberMean
print healthy_edgeNumberStd,disease_edgeNumberStd
t=ttest_ind(healthy_edge_list,disease_edge_list,equal_var = True)
print "--------"
print "t score:"
print t
print "--------"




#plot the difference between patient and disease brain network for small scale
##############################################################average path length
N=1
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
fig, (ax1,ax2) = plt.subplots(2,1)
pl.title('Comparison between Healthy and Disease Group')

healthy_groupMeans=[healthy_aplMean]
healthy_groupStd=[healthy_aplStd]
rects1 = ax1.bar(ind, healthy_groupMeans, width, color='r', yerr=healthy_groupStd)

disease_groupMeans=[disease_aplMean]
disease_groupStd=[disease_aplStd]
rects2 = ax1.bar(ind+width, disease_groupMeans, width, color='y', yerr=disease_groupStd)

ax1.set_ylabel('Counts')
ax1.set_xticks(ind+width*2)
ax1.set_xticklabels( ("Average Path Length") )
ax1.legend( (rects1[0], rects2[0]), ("healthy", "eye disease") )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()

#############################################clustering coefficient
healthy_groupMeans=[healthy_ccMean]
healthy_groupStd=[healthy_ccStd]
rects1 = ax2.bar(ind, healthy_groupMeans, width, color='r', yerr=healthy_groupStd)

disease_groupMeans=[disease_ccMean]
disease_groupStd=[disease_ccStd]
rects2 = ax2.bar(ind+width, disease_groupMeans, width, color='y', yerr=disease_groupStd)

ax2.set_ylabel('Counts')
ax2.set_xticks(ind+width*2)
ax2.set_xticklabels( ("Clustering Coefficient") )
ax.legend( (rects1[0], rects2[0]), ("healthy", "eye disease") )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)


plt.show()