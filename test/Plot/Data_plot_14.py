# -*- coding: utf-8 -*-
"""
Created on Sat Mar 01 17:31:48 2014

@author: wu
"""
import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import test.Global.globalvars as vars
import numpy as np

selection_percentile=range(0,105,5)
#selection_percentile=[0,100]
apl_list=[]
transitivity_global_list=[]
transitivity_local_list=[]
for p in selection_percentile:
    vars.PERCENTILE=p
    print "used to be set: %s" %vars.PERCENTILE
    selector=Selector("all",weight_filter_method="binarilize")
    data=selector.data
    table_list=[]
    for file_name,g in data:
        measures=NA(g).extractInfo()[["apl","transitivity_global","transitivity_local"]]
        table_list.append(measures)
    comtable=pd.concat(table_list).mean()
    apl_list.append(comtable["apl"])
    transitivity_global_list.append(comtable["transitivity_global"])
    transitivity_local_list.append(comtable["transitivity_local"])
#plot
fig=pl.figure()
fig.subplots_adjust(hspace=0.6)
"""
pl.subplot(3,1,1)
pl.title("The Influence of Graph Pruning on Global Transitivity Over All Patients and Time Stamps")
pl.xlabel("Percentage Being Pruned")
pl.ylabel("Global Transitivity")
pl.plot(selection_percentile,transitivity_global_list)
ax=pl.gca()
major_locator=plt.MultipleLocator(10)
minor_locator=plt.MultipleLocator(5)
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)    
"""
pl.subplot(2,1,1)
pl.title("The Influence of Graph Pruning on Average Path Length Over All Patients and Time Stamps")
pl.xlabel("Percentage Being Pruned")
pl.ylabel("Average Path Length")
pl.plot(selection_percentile,apl_list)
ax=pl.gca()
major_locator=plt.MultipleLocator(10)
minor_locator=plt.MultipleLocator(5)
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)

pl.subplot(2,1,2)
pl.title("The Influence of Graph Pruning on local transitivity Over All Patients and Time Stamps")
pl.xlabel("Percentage Being Pruned")
pl.ylabel("Local Transitivity")
pl.plot(selection_percentile,transitivity_local_list)
ax=pl.gca()
major_locator=plt.MultipleLocator(10)
minor_locator=plt.MultipleLocator(5)
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)
    
pl.show()