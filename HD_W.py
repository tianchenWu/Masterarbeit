# -*- coding: utf-8 -*-
"""
Created on Sun Feb 09 22:47:28 2014

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
healthy_hub_list,disease_hub_list=[],[]

#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,0.5)    
    edgeNumber=get_edge_number(graph)
    print edgeNumber
    healthy_edge_list.append(edgeNumber)
    healthy_apl_list.append(get_average_path_length(graph))
    healthy_cc_list.append(avg_wcc(graph))
    healthy_hub_list.append(max(graph.betweenness(directed=False,weights="weight")))
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)

print "-----------------------------------------------------"

#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.5)
    #print graph.es["weight"]
    edgeNumber=get_edge_number(graph)
    disease_edge_list.append(edgeNumber)
    #print edgeNumber
    disease_apl_list.append(get_average_path_length(graph))
    disease_cc_list.append(avg_wcc(graph))
    disease_hub_list.append(max(graph.betweenness(directed=False,weights="weight")))
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)

#average over the number of patients in the group, 15
numPatient=15
healthy_edgeNumberMean,disease_edgeNumberMean=np.mean(np.asarray(healthy_edge_list)),np.mean(np.asarray(disease_edge_list))
healthy_aplMean,disease_aplMean=np.mean(np.asarray(healthy_apl_list)),np.mean(np.asarray(disease_apl_list))
healthy_ccMean,disease_ccMean=np.mean(np.asarray(healthy_cc_list)),np.mean(np.asarray(disease_cc_list))
healthy_hubMean,disease_hubMean=np.mean(np.asarray(healthy_hub_list)),np.mean(np.asarray(disease_hub_list))

healthy_edgeNumberStd,disease_edgeNumberStd=np.std(np.asarray(healthy_edge_list)),np.std(np.asarray(disease_edge_list))
healthy_aplStd,disease_aplStd=np.std(np.asarray(healthy_apl_list)),np.std(np.asarray(disease_apl_list))
healthy_ccStd,disease_ccStd=np.std(np.asarray(healthy_cc_list)),np.std(np.asarray(disease_cc_list))
healthy_hubStd,disease_hubStd=np.std(np.asarray(healthy_hub_list)),np.std(np.asarray(disease_hub_list))




#EX1
print healthy_edgeNumberMean,disease_edgeNumberMean
print healthy_edgeNumberStd,disease_edgeNumberStd
t=ttest_ind(healthy_edge_list,disease_edge_list,equal_var = True)
print "--------"
print "t score:"
print t
print "--------"






print "edge number: (%s,%s)" %(healthy_edgeNumberMean,disease_edgeNumberMean)       
print "average path length: (%s,%s)" %(healthy_aplMean,disease_aplMean)
print "clustering coefficient (%s,%s)" %(healthy_ccMean,disease_ccMean)
print "Kleinberg hub score: (%s,%s)" %(healthy_hubMean,disease_hubMean)








"""
CONCLUSION:

19.4879791622 18.4321886222
7.4381325253 8.27184565602
--------
t score:
(array(0.355116215254789), 0.72516363151482222)
--------
edge number: (19.4879791622,18.4321886222)
average path length: (7.63373342741,8.2210089858)
clustering coefficient (0.303205560906,0.295985804949)
Kleinberg hub score: (0.604582530533,0.618292524794)


"""


#simulate random graph
#run=10000
#timeseries_length=150
run=10
timeseries_length=200
e_,apl_,cc_,hub_=[],[],[],[]

for i in range(run):
    sim_graphlist=[]
    for j in range(timeseries_length):
        graph=simulateWeightedRandomGraph(0.5)
        #print graph.is_weighted()
        #print graph.es["weight"]
        sim_graphlist.append(graph)
    graph=average_graph(sim_graphlist,0.5)
    e_.append(get_edge_number(graph))
    apl_.append(get_average_path_length(graph))
    cc_.append(avg_wcc(graph))
    hub_.append(max(graph.betweenness(directed=False,weights="weight")))
print e_
avg_e,std_e=np.mean(np.asarray(e_)),np.std(np.asarray(e_))
avg_apl,std_apl=np.mean(np.asarray(apl_)),np.std(np.asarray(apl_))
avg_cc,std_cc=np.mean(np.asarray(cc_)),np.std(np.asarray(cc_))
avg_hub,std_hub=np.mean(np.asarray(hub_)),np.std(np.asarray(hub_))

print "Random Graph Information------------------------------------------"
print avg_e,std_e
print avg_apl,std_apl
print avg_cc,std_cc
print avg_hub,std_hub








#-------------------------------------------------------------------------------------


#plot the difference between random and human brain network for small scale

N=3
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
fig, ax = plt.subplots()

healthy_groupMeans=[healthy_aplMean,healthy_ccMean,healthy_hubMean]
healthy_groupStd=[healthy_aplStd,healthy_ccStd,healthy_hubStd]
rects1 = ax.bar(ind, healthy_groupMeans, width, color='r', yerr=healthy_groupStd)

disease_groupMeans=[disease_aplMean,disease_ccMean,disease_hubMean]
disease_groupStd=[disease_aplStd,disease_ccStd,disease_hubStd]
rects2 = ax.bar(ind+width, disease_groupMeans, width, color='y', yerr=disease_groupStd)


random_groupMeans=[avg_apl,avg_cc,avg_hub]
random_groupStd=[std_apl,std_cc,std_hub]
rects3 = ax.bar(ind+width*2, random_groupMeans, width, color='b', yerr=random_groupStd)
# add some
ax.set_ylabel('Counts')
ax.set_title('Comparison between Healthy and Disease Group')
ax.set_xticks(ind+width*2)
ax.set_xticklabels( ("Average Path Length", "Clustering Coefficient", "Max Betweenness") )
ax.legend( (rects1[0], rects2[0],rects3[0]), ("healthy", "eye disease","random graph") )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
