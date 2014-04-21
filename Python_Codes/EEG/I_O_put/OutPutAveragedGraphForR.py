# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:41:57 2014

@author: wu
"""



# -*- coding: utf-8 -*-
"""
Created on Sat Feb 08 22:08:30 2014

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
from test.Global.HelpFunction import parse_file_name

#marke the relevant nodes as 1, others as 0
#[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1]

patientsNetworkDir="C:\\Users\\wu\\Desktop\\Patient"

selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data
healthy_data,disease_data=data

#healthy group
for file_name,eeg in healthy_data:
    patientPath=patientsNetworkDir+"\\"+"Healthy"
    if not os.path.isdir(patientPath):
        os.makedirs(patientPath)
    graphPath=patientPath+"\\"+parse_file_name(file_name).split("_")[1]+".net"
    print graphPath
    g=average_graph(eeg.graphlist,0.3,isWeighted=False)
    g.write_pajek(graphPath)



#eye disease group
for file_name,eeg in disease_data:
    graph=average_graph(eeg.graphlist,0.3,isWeighted=False)
    patientPath=patientsNetworkDir+"\\"+"Disease"
    if not os.path.isdir(patientPath):
        os.makedirs(patientPath)
    graphPath=patientPath+"\\"+parse_file_name(file_name).split("_")[1]+".net"
    print patientPath
    graph.write_pajek(graphPath)



#a=ig.Graph.Read_Pajek("C:\\Users\\wu\\Desktop\\Patient\\alpha_AM_prae_closed1.pajek")
#layout=a.layout("kk")
#ig.plot(a,layout=layout)
