# -*- coding: utf-8 -*-
"""
Created on Sun Feb 02 14:29:24 2014

@author: wu
"""

from test.I_O_put.Selector import *
import test.Global.globalvars as vars
from test.StatsAnalysis.GraphMeasuresTable import *
import pandas as pd
import matplotlib.pylab as pl
from test.StatsAnalysis.GraphMeasuresTable import *
from test.Global.HelpFunction import parse_file_name


theo_p_list=np.linspace(0.05,0.95,19)#theoretical p list

###########################################################
#For simulated Graphs
###########################################################
#read cvs into dataframe
#p_table_list=[]
#iterate p_table_list, plot histogram for each p_table
#for p in p_table_list:
#    pass

path="C:\\Users\\wu\\Desktop\\WeightedRandomGraphSimulation\\Tables\\p_0.5.csv"
data=pd.DataFrame.from_csv(path)
#print data
x=data.ix[4,"apl"]
print x
#print x.mean()[0]
#print x.std()[0]
print x.ix(6)

#x.hist(grid=True)
###########################################################

"""
############################################################
#For Patients
############################################################
#calculate weighted average path length and weighted local transitivity for each patient on every time stamp
data=Selector("all",weight_filter_method=vars.weight_filter_method).data
for (file_name,g) in data:
    table=NA(g).extractInfo()[["transitivity_local","apl"]]
    p_list=g.measures
    assert p_list.shape[1]==1,"weighted random graph model should be used"
    p_list=[entry[0] for entry in p_list]#convert to a list of p
    #extract corresponding weighted avg and local transitivity from table by simulated graphs
    apl_sd_list,cc_sd_list,real_apl_list,real_cc_list,cc_mean_list,apl_mean_list=[],[],[],[],[],[]
    for i in range(len(p_list)):
        p=p_list[i]
        #find the nearst p in theo_p_list
        theo_p=min(theo_p_list, key=lambda x:abs(x-p))
        #find the simulated mean and sd for apl,local_transitivity
        path="C:\\Users\\wu\\Desktop\\WeightedRandomGraphSimulation\\Tables\\p_"+str(theo_p)+".csv"
        p_table=pd.DataFrame.from_csv(path)
        apl_mean=p_table[["apl"]].mean()[0]
        apl_mean_list.append(apl_mean)
        apl_sd=p_table[["apl"]].std()[0]
        apl_sd_list.append(apl_sd)
        cc_mean=p_table[["local_transitivity"]].mean()[0]
        cc_mean_list.append(cc_mean)
        cc_sd=p_table[["local_transitivity"]].std()[0]
        cc_sd_list.append(cc_sd)
        real_cc=table.ix[i,"transitivity_local"]
        real_cc_list.append(real_cc)
        real_apl=table.ix[i,"apl"]
        real_apl_list.append(real_apl)
    #plot! weighted average path length,weighted local transitivity
    fig,(ax1,ax2)=pl.subplots(1,2,sharex=True)
    fig.set_size_inches(18.5,10.5)
    x=range(len(apl_mean_list))
    ax1.plot(x,apl_mean_list,lw=5,label="simulated apl",color="yellow")
    ax1.plot(x,real_apl_list,lw=1,label="empirical apl",color="blue")
    apl_lower_bound=[apl_mean_list[i]-apl_sd_list[i] for i in x]
    apl_upper_bound=[apl_mean_list[i]+apl_sd_list[i] for i in x]
    ax1.fill_between(x,apl_lower_bound,apl_upper_bound,facecolor="yellow",alpha=0.5)
    ax1.legend()
    ax1.set_title("Weighted Average Path Length")
    ax1.set_xlabel("Time Series")
    #ax1.set_ylabel("Weighted Average Path Length")
    ax1.grid()
    
    ax2.plot(x,cc_mean_list,lw=5,label="simulated clustering coefficient",color="yellow")
    ax2.plot(x,real_cc_list,lw=1,label="empirical clustering coefficient",color="blue")
    cc_lower_bound=[cc_mean_list[i]-cc_sd_list[i] for i in x]
    cc_upper_bound=[cc_mean_list[i]+cc_sd_list[i] for i in x]
    ax2.fill_between(x,cc_lower_bound,cc_upper_bound,facecolor="yellow",alpha=0.5)
    ax2.legend()
    ax2.set_title("Weighted Clustering Coefficient")
    ax2.grid()
    #ax1.set_ylabel("Weighted Clustering Coefficient")
    patientID=parse_file_name(file_name)
    fig.suptitle(patientID)
    #save picture into file
    save_path="C:\\Users\\wu\\Desktop\\WeightedRandomGraphSimulation\\PlotAgainstPatient"+"\\"+patientID+".PNG"
    savefig(save_path,dpi=100)
    pl.clf()
    
"""
    
        
###########################################################

    

def weightedRandomGraphFilter():
    pass

   