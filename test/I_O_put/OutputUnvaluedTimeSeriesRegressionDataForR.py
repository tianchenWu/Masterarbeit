# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 13:57:48 2014

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

patientsNetworkDir="C:\\Users\\wu\\Desktop\\Patient"

selector=Selector("all",weight_filter_method="binarilize")
#print len(selector.data)
cv=selector.build_clinical_variables()
cv=[float(x) for x in cv]
print cv
#print selector.data
filenames=[]
for file_name,eeg in selector.data:
    patientPath=patientsNetworkDir+"\\"+parse_file_name(file_name).split("_")[1]
    print parse_file_name(file_name).split("_")[1]
    filenames.append(parse_file_name(file_name).split("_")[1])
    #print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".net"
        g.write_pajek(patientStampPath)
    
"""
data=zip(filenames,cv)
data=np.asarray(data)
cv_table=pd.DataFrame(data)
print cv_table[1:20]
cv_table.to_csv("C:\Users\wu\Desktop\R Scripts\\cv_table.txt")
"""