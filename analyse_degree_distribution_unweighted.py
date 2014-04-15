# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:28:33 2014

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
patientsNetworkDir="C:\\Users\\wu\\Desktop\\Patient"

sampleSize=3
#healthy group
for file_name,eeg in healthy_data:
    #for graph in eeg.graphlist[1:sampleSize]:
    #    degrees=graph.degree()
    #    fig,ax=pl.subplots()
    #    ax.hist(degrees,bins=10)
        
    graph=average_graph(eeg.graphlist,0.2)
    
    patientPath=patientsNetworkDir+"\\"+"Healthy_"+parse_file_name(file_name)
    print patientPath
    graphPath=patientPath+".pajek"
    g.write_pajek(graphPath)
        
    #degrees=graph.degree()
    #fig,ax=pl.subplots()
    #ax.hist(degrees,bins=10)
    
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)


    
#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.2)
    
    patientPath=patientsNetworkDir+"\\"+"Disease_"+parse_file_name(file_name)
    print patientPath
    graphPath=patientPath+".pajek"
    g.write_pajek(graphPath)
    
    #graph=average_graph(eeg.graphlist,0.2)
    #degrees=graph.degree()
    #fig,ax=pl.subplots()
    #ax.hist(degrees,bins=10)
    
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)

