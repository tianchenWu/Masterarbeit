

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

for file_name,eeg in healthy_data:
    patientPath=patientsNetworkDir+"\\"+"Healthy\\"+parse_file_name(file_name).split("_")[1]
    print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".net"
        g.write_pajek(patientStampPath)
        


#eye disease group
for file_name,eeg in disease_data:
    patientPath=patientsNetworkDir+"\\"+"Disease\\"+parse_file_name(file_name).split("_")[1]
    print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".net"
        g.write_pajek(patientStampPath)

