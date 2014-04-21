# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 16:19:23 2014

@author: wu
"""
from test.I_O_put.Selector import *
import test.Global.globalvars as vars
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
from test.Global.HelpFunction import parse_file_name
from test.StatsAnalysis.NetworkSimulation import simulateWeightedRandomGraph

run=1000
#calculate weighted average path length and weighted local transitivity for each patient on every time stamp
data=Selector("2",weight_filter_method=vars.weight_filter_method).data
for (file_name,g) in data:
    table=NA(g).extractInfo()[["transitivity_local","apl"]]
    p_list=g.measures
    assert p_list.shape[1]==1,"weighted random graph model should be used"
    p_list=[entry[0] for entry in p_list]#convert to a list of p
    #extract corresp onding weighted avg and local transitivity from table by simulated graphs
    swi_list=[]
    for i in range(len(p_list)):
        p=p_list[i]
        apl_list,cc_list=[],[]
        for j in range(run):
            g=simulateWeightedRandomGraph(p)
            apl_list.append(get_average_path_length(g))
            cc_list.append(get_clustering_coefficient(g))
        avg_apl=np.mean(np.asarray(apl_list))
        avg_cc=np.mean(np.asarray(cc_list))
        real_cc=table.ix[i,"transitivity_local"]
        real_apl=table.ix[i,"apl"]
        
        ccNormalized=float(real_cc)/float(avg_cc)
        aplNormalized=float(real_apl)/float(avg_apl)
        swi=ccNormalized/aplNormalized
        swi_list.append(swi)
    
    #plot! weighted average path length,weighted local transitivity
    fig,ax=pl.subplots()
    fig.set_size_inches(18.5,10.5)
    x=range(len(swi_list))
    ax.plot(x,swi_list,lw=2,label="Small World Index",color="blue")
    ax.axhline(y=1,color="r")
    ax.set_title("Small World Properties along Time Series")
    ax.set_xlabel("Time Series")
    ax.set_ylabel("Small World Index")
    ax.grid(True)
