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
from test.Global.HelpFunction import average_graph
import igraph as ig
from scipy.stats import ttest_ind

selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data

healthy_data,disease_data=data

healthy_edge_list,disease_edge_list=[],[]
healthy_apl_list,disease_apl_list=[],[]
healthy_cc_list,disease_cc_list=[],[]
healthy_motif3_list,disease_motif3_list=[],[]
healthy_motif4_list,disease_motif4_list=[],[]
healthy_numClique_list,disease_numClique_list=[],[]
healthy_hub_list,disease_hub_list=[],[]
healthy_hubMax_list,disease_hubMax_list=[],[]

#healthy group
for file_name,eeg in healthy_data:
    graph=average_graph(eeg.graphlist,0.3331)
    edgeNumber=get_edge_number(graph)
    healthy_edge_list.append(edgeNumber)
    print edgeNumber
    healthy_apl_list.append(get_average_path_length(graph))
    healthy_cc_list.append(get_clustering_coefficient(graph))
    healthy_motif3_list.append(get_motif3(graph))
    healthy_motif4_list.append(get_motif4(graph))
    healthy_numClique_list.append(get_numClique(graph))
    healthy_hub_list.append(get_hub_score(graph))
    healthy_hubMax_list.append(get_max_hub_score(graph))
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)

print "-----------------------------------------------------"
    
#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.3)
    edgeNumber=get_edge_number(graph)
    disease_edge_list.append(edgeNumber)
    print edgeNumber
    disease_apl_list.append(get_average_path_length(graph))
    disease_cc_list.append(get_clustering_coefficient(graph))
    disease_motif3_list.append(get_motif3(graph))
    disease_motif4_list.append(get_motif4(graph))
    disease_numClique_list.append(get_numClique(graph))
    disease_hub_list.append(get_hub_score(graph))
    disease_hubMax_list.append(get_max_hub_score(graph))
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)

#average over the number of patients in the group, 15
numPatient=15
healthy_edgeNumberMean,disease_edgeNumberMean=np.mean(np.asarray(healthy_edge_list)),np.mean(np.asarray(disease_edge_list))
healthy_aplMean,disease_aplMean=np.mean(np.asarray(healthy_apl_list)),np.mean(np.asarray(disease_apl_list))
healthy_ccMean,disease_ccMean=np.mean(np.asarray(healthy_cc_list)),np.mean(np.asarray(disease_cc_list))
healthy_motif3Mean,disease_motif3Mean=np.mean(np.asarray(healthy_motif3_list)),np.mean(np.asarray(disease_motif3_list))
healthy_motif4Mean,disease_motif4Mean=np.mean(np.asarray(healthy_motif4_list)),np.mean(np.asarray(disease_motif4_list))
healthy_numCliqueMean,disease_numCliqueMean=np.mean(np.asarray(healthy_numClique_list)),np.mean(np.asarray(disease_numClique_list))
healthy_hubMean,disease_hubMean=np.mean(np.asarray(healthy_hub_list)),np.mean(np.asarray(disease_hub_list))
healthy_hubMaxMean,disease_hubMaxMean=np.mean(np.asarray(healthy_hubMax_list)),np.mean(np.asarray(disease_hubMax_list))


healthy_edgeNumberStd,disease_edgeNumberStd=np.std(np.asarray(healthy_edge_list)),np.std(np.asarray(disease_edge_list))
healthy_aplStd,disease_aplStd=np.std(np.asarray(healthy_apl_list)),np.std(np.asarray(disease_apl_list))
healthy_ccStd,disease_ccStd=np.std(np.asarray(healthy_cc_list)),np.std(np.asarray(disease_cc_list))
healthy_motif3Std,disease_motif3Std=np.std(np.asarray(healthy_motif3_list)),np.std(np.asarray(disease_motif3_list))
healthy_motif4Std,disease_motif4Std=np.std(np.asarray(healthy_motif4_list)),np.std(np.asarray(disease_motif4_list))
healthy_numCliqueStd,disease_numCliqueStd=np.std(np.asarray(healthy_numClique_list)),np.std(np.asarray(disease_numClique_list))
healthy_hubStd,disease_hubStd=np.std(np.asarray(healthy_hub_list)),np.std(np.asarray(disease_hub_list))
healthy_hubMaxStd,disease_hubMaxStd=np.std(np.asarray(healthy_hubMax_list)),np.std(np.asarray(disease_hubMax_list))




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
print "motif3: (%s,%s)" %(healthy_motif3Mean,disease_motif3Mean)
print "motif4: (%s,%s)" %(healthy_motif4Mean,disease_motif4Mean)
print "size of largest clique: (%s,%s)" %(healthy_numCliqueMean,disease_numCliqueMean)
print "Kleinberg hub score: (%s,%s)" %(healthy_hubMean,disease_hubMean)
print "Kleinberg hub score max: (%s,%s)" %(healthy_hubMaxMean,disease_hubMaxMean)







"""
CONCLUSION:

EX1:
DES:
EX1: IF THE NUMBER OF EDGE IN HEALTHY GROUP SIGNIFICANTLY DIFFERS FROM DISEASE GROUP

threshold 0.2: std healthy:disease=20.2638155889:20.5290038726
threshold 0.5: std healthy:disease=17.3238435561:18.8484010521
threshold 0.8: std healthy:disease=7.12616462217:7.03767638421
No significant difference on std so we do not use welch't test

threshold 0.5: t=0.8477077539966729, p-value=0.40378909532216745
can not reject the null hypothesis(two group have equal edge number)

threshold 0.2: t=1.1328336795219374, p-value=0.2668917704878328
can not reject the null hypothesis(two group have equal edge number)

threshold 0.8: t=0.672453103779989, p-value=0.50680710994511313
can not reject the null hypothesis(two group have equal edge number)

EX2: IF THESE TWO GROUP DIFFERS FROM RANDOM GRAPH

to control the number of edges, we set
On healthy threshold 0.535
On disease threshold 0.5

Result:
edge number: (32.0,32.0666666667)
average path length: (10.1384015595,10.538791423)
clustering coefficient (0.379877218298,0.370869111395)
motif3: (96.4666666667,98.8)
motif4: (262.666666667,276.866666667)
size of largest clique: (5.2,4.8)
Kleinberg hub score: (0.395638265745,0.379464649636)

Compared with random graph(mean,std):
32.0 0.0
3.01653567251 0.977586869605
0.163767125769 0.0675757726166
87.4614 5.92531096568
257.6046 31.1955871693
3.0973 0.303039122887
0.498857073004 0.0574042646092

They differ on  [average path length,clustering coefficient, motif3,hub score] shows a small world


On healthy threshold 0.217
On disease threshold 0.2

edge number: (131.133333333,131.6)
average path length: (1.36647173489,1.37504873294)
clustering coefficient (0.820022084325,0.819274404646)
motif3: (778.266666667,772.266666667)
motif4: (3324.66666667,3239.0)
size of largest clique: (11.3333333333,11.8)
Kleinberg hub score: (0.836334806327,0.838001250103)
Random Graph Information------------------------------------------
131.0 0.0
1.23391812866 1.20792265079e-13
0.765954173117 0.0107133104278
836.4258 6.67603882254
3682.2151 30.4399085411
8.3003 0.643521491483
0.840215337406 0.0268231200957

Then the phenomenon of small world is no obvious anymore.


On healthy threshold 0.3331
On disease threshold 0.3

t score:
(array(0.049132486269276784), 0.96116251845337342)
--------
edge number: (80.2,79.8)
average path length: (2.64873294347,2.85925925926)
clustering coefficient (0.652612821761,0.625991318623)
motif3: (389.333333333,376.6)
motif4: (1564.06666667,1461.13333333)
size of largest clique: (8.06666666667,7.93333333333)
Kleinberg hub score: (0.649653998547,0.662872502206)
Random Graph Information------------------------------------------
80.0 0.0
1.53786900585 0.00696472480979
0.468071281846 0.031898083908
437.5232 7.80979268355
2067.0006 54.69267958
4.9789 0.460494071623
0.708972639623 0.0443821622138



"""

#simulate random graph
run=10000
e_,apl_,cc_,motif3_,motif4_,numClique_,hub_,hubMax_=[],[],[],[],[],[],[],[]

for i in range(run):
    graph=ig.Graph.Erdos_Renyi(n=vars.N,m=80, directed=False, loops=False)
    e_.append(get_edge_number(graph))
    apl_.append(get_average_path_length(graph))
    cc_.append(get_clustering_coefficient(graph))
    motif3_.append(get_motif3(graph))
    motif4_.append(get_motif4(graph))
    numClique_.append(get_numClique(graph))
    hub_.append(get_hub_score(graph))
    hubMax_.append(get_max_hub_score(graph))

avg_e,std_e=np.mean(np.asarray(e_)),np.std(np.asarray(e_))
avg_apl,std_apl=np.mean(np.asarray(apl_)),np.std(np.asarray(apl_))
avg_cc,std_cc=np.mean(np.asarray(cc_)),np.std(np.asarray(cc_))
avg_motif3,std_motif3=np.mean(np.asarray(motif3_)),np.std(np.asarray(motif3_))
avg_motif4,std_motif4=np.mean(np.asarray(motif4_)),np.std(np.asarray(motif4_))
avg_numClique,std_numClique=np.mean(np.asarray(numClique_)),np.std(np.asarray(numClique_))
avg_hub,std_hub=np.mean(np.asarray(hub_)),np.std(np.asarray(hub_))
avg_hubMax,std_hubMax=np.mean(np.asarray(hubMax_)),np.std(np.asarray(hubMax_))



print "Random Graph Information------------------------------------------"
print avg_e,std_e
print avg_apl,std_apl
print avg_cc,std_cc
print avg_motif3,std_motif3
print avg_motif4,std_motif4
print avg_numClique,std_numClique
print avg_hub,std_hub
print avg_hubMax,std_hubMax








#-------------------------------------------------------------------------------------

#plot the difference between random and human brain network for large scale

N = 3

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
fig, ax = plt.subplots()

healthy_groupMeans=[healthy_edgeNumberMean,healthy_motif3Mean,healthy_motif4Mean]
healthy_groupStd=[healthy_edgeNumberStd,healthy_motif3Std,healthy_motif4Std]
rects1 = ax.bar(ind, healthy_groupMeans, width, color='r', yerr=healthy_groupStd)

disease_groupMeans=[disease_edgeNumberMean,disease_motif3Mean,disease_motif4Mean]
disease_groupStd=[disease_edgeNumberStd,disease_motif3Std,disease_motif4Std]
rects2 = ax.bar(ind+width, disease_groupMeans, width, color='y', yerr=disease_groupStd)


random_groupMeans=[avg_e,avg_motif3,avg_motif4]
random_groupStd=[std_e,std_motif3,std_motif4]
rects3 = ax.bar(ind+width*2, random_groupMeans, width, color='b', yerr=random_groupStd)
# add some
ax.set_ylabel('Counts')
ax.set_title('Comparison between Healthy and Disease Group')
ax.set_xticks(ind+width*2)
ax.set_xticklabels( ("Edge", "motif 3", "motif 4") )
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

#plot the difference between random and human brain network for small scale

N=5
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
fig, ax = plt.subplots()

healthy_groupMeans=[healthy_aplMean,healthy_ccMean,healthy_numCliqueMean,healthy_hubMean,healthy_hubMaxMean]
healthy_groupStd=[healthy_aplStd,healthy_ccStd,healthy_numCliqueStd,healthy_hubStd,healthy_hubMaxStd]
rects1 = ax.bar(ind, healthy_groupMeans, width, color='r', yerr=healthy_groupStd)

disease_groupMeans=[disease_aplMean,disease_ccMean,disease_numCliqueMean,disease_hubMean,disease_hubMaxMean]
disease_groupStd=[disease_aplStd,disease_ccStd,disease_numCliqueStd,disease_hubStd,disease_hubMaxMean]
rects2 = ax.bar(ind+width, disease_groupMeans, width, color='y', yerr=disease_groupStd)


random_groupMeans=[avg_apl,avg_cc,avg_numClique,avg_hub,avg_hubMax]
random_groupStd=[std_apl,std_cc,std_numClique,std_hub,std_hubMax]
rects3 = ax.bar(ind+width*2, random_groupMeans, width, color='b', yerr=random_groupStd)
# add some
ax.set_ylabel('Counts')
ax.set_title('Comparison between Healthy and Disease Group')
ax.set_xticks(ind+width*2)
ax.set_xticklabels( ("Average Path Length", "Clustering Coefficient", " Clique", "Kleinburg Hub Score","Kleinburg Hub Score Max") )
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
