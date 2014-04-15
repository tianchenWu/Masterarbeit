# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:08:27 2014

@author: wu
"""

import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import test.Global.globalvars as vars
import numpy as np

###################################################################################
"""
Plot and save the graph measures of patient
"""
"""
def plot_apl_cc(file_name,EEGGraph):

    measures=NA(EEGGraph).extractInfo()[["apl","transitivity_global"]]
    measures.boxplot()
    
    save_path="C:\\Users\\wu\\Desktop\\hi"+"\\"+file_name+".jpg"
    savefig(save_path)
    pl.clf()
    
if __name__=="__main__":
   selector=Selector("3",weight_filter_method="binarilize")
   data=selector.data
   for file_name,g in data:
       name=file_name.split("\\")[-1]
       plot_apl_cc(name,g)
"""


###################################################################################
"""
Search percentile cut for small world properties
"""


def plot_apl_cc2(EEGGraph):

    measures=NA(EEGGraph).extractInfo()[["apl","transitivity_global"]].mean()
    print measures
    
    #save_path="C:\\Users\\wu\\Desktop\\hi"+"\\"+file_name+".jpg"
    #savefig(save_path)
    #pl.clf()




#################################################################################
"""
Search small world K

if __name__=="__main__":
    selector=Selector("all",weight_filter_method="binarilize")
    data=selector.data
    cv=selector.build_clinical_variables()
    mean_degrees=[]
    for file_name,g in data:
        mean_ds=[x.mean() for x in g.ds]
        mean=np.asarray(mean_ds).mean()
        mean_degrees.append(mean)
        #pl.figure()
        #pl.plot(mean_ds)
        #pl.xlabel("Time Series")
        #pl.ylabel("Average Degree(K)")
    cv=[nested[0] for nested in cv]
    print cv
    pl.figure()
    pl.scatter(cv,mean_degrees)


"""
#################################################################################

"""
Plotting degree sequence

For each patient, we get a list of 19 values, each of which is medianed on time series

"""



"""
#Plot averaged graph measures against clinical variables
selector=Selector("all",weight_filter_method=vars.weight_filter_method)
data=selector.data
Y=selector.build_clinical_variables()
measure_list=[]
for file_name,g in data:
    #p_list=[measure[0] for measure in g.measures]
    #p_mean=np.mean(np.asarray(p_list))
    mean_measure=g.measures.mean(axis=0)
    measure_list.append(mean_measure)

X1,X2,X3=np.swapaxes(np.asarray(measure_list),0,1)

fig = pl.figure(1, figsize=(18, 16))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X1, X2, X3,s=80, c=Y,cmap=pl.cm.Paired)
ax.set_title("Small World")
ax.set_xlabel("Density")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Clustering Coefficient")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Average Path Length")
ax.w_zaxis.set_ticklabels([])          
    
#fig,ax=pl.subplots(1)
#ax.scatter(X,cv)

"""






















            
            