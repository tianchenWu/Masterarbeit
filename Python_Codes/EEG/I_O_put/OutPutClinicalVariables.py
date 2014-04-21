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


all_names=["ID","HRP_DA_g","HRP_DA","HRP_DA_CMF","HRP_DA_g_CMF","SP_FA","SP_FT","SP_MT","KP_ME","KP_AE","VA_N_dec","VA_F_dec","VA_N_log","VA_F_log"]
central_vision=["VA_N_log","VA_F_log","SP_FT"]
#central_vision=["SP_FT"]
cv_names=central_vision
"""
Single variable
"""
"""
selector=Selector("all",weight_filter_method="discretize")

cv=selector.build_clinical_variables()
cv=[float(x) for x in cv]
print cv

cv_table=pd.DataFrame(data)
print cv_table[1:20]
cv_table.to_csv("C:\Users\wu\Desktop\R Scripts\\cv_table.txt")
"""

"""
Multiple variables
"""
data=[]
for cv in central_vision:
    vars.clinical_variable_names=[cv]
    selector=Selector("all",weight_filter_method="binarilize")
    #print len(selector.data)
    #print vars.clinical_variable_names
    cv=selector.build_clinical_variables()
    cv=[float(x) for x in cv]
    #print cv
    data.append(cv)

#name for each row
key=[]
for file_name,_ in selector.data:
    print parse_file_name(file_name).split("_")[1]
    k=parse_file_name(file_name).split("_")[1]
    key.append(k)


data=np.asarray(data)
data=np.asarray(data).swapaxes(1,0)  
cv_table=pd.DataFrame(data)
cv_table.columns=central_vision
cv_table["KEY"]=key  
print cv_table[1:20]

cv_table.to_csv("C:\\Users\\wu\\spyder_workspace\\HurryUp2\\TempData\\cv_table.txt")