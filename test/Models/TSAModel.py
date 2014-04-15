# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:08:14 2014

@author: wu
"""

from test.I_O_put.ReadDGS import ReadDGS
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.vector_ar.var_model import *
import statsmodels.tsa.vector_ar.plotting as plotting
import pandas as pd
from statsmodels.tsa.ar_model import *
from test.I_O_put.Selector import Selector
import gzip
import random
import pylab
import test.Filter.LowPassFilter as lpf
import test.Global.globalvars as vars
from test.Filter.LowPassFilter import savitzky_golay
from sklearn import linear_model
import statsmodels.tsa.stattools as sts
import matplotlib.pylab as pl

class TSAModel:
    """
    model_name: can be "AR"
    """
    def __init__(self,data,model_name):
        self.data=data
        self.model_name=model_name

    def get_graph_measures(self,g):
        """
        Input: EEG graph
        Output:nxm ndarray, n is the length of time series, m is the number of measures
        """
        return g.measures
        
    def build_model(self):
        """
        input:EEG data in the form of data=[(file_name,g),.....]
        output: var_parameters, namely regressors as 2d-array
        """
        var_parameters=[]
        if self.model_name=="AR":
            var_parameters=self.build_AR_model()
        if self.model_name=="VAR":
            var_parameters=self.build_VAR_model()
        if self.model_name=="SM":
            var_parameters=self.sample_model()
        return var_parameters
    
    def build_VAR_model(self):
        var_parameters=[]
        print self.data[1]        
        for file_name,g in self.data:    
            raw_data=self.get_graph_measures(g)
            assert raw_data.shape[1]>1,"VAR can not be used, please use AR model"
            
            ###############################################################################filter in time series model
            #group the data by measure
            original_raw_data=raw_data.swapaxes(0,1);
            raw_data=[]
            #the form of data_each_measure is 1d numpy array
            for data_each_measure in original_raw_data:
                filtered_data_each_measure=savitzky_golay(data_each_measure, window_size=31, order=12)
                raw_data.append(filtered_data_each_measure)
            raw_data=np.asarray(raw_data).swapaxes(1,0)
            #print raw_data
            ################################################################################
            columns=vars.graph_measurenames
            #a switch to control if loop will be exectued maximal once
            if vars.graph_measurenames[0]=="degree sequence":
                columns=g.vertex_name
                
            if vars.graph_measurenames[0]=="TWS":
                columns=["K","p"]
            if vars.graph_measurenames[0]=="GCOR":
                columns=map(str,range(361))
            #prepare input data for VAR model
            #time series model put stricts on index, it must exist and has to be DatatimeIndex
            index=pd.date_range("1",periods=len(raw_data))
            raw_data = pd.DataFrame(raw_data,index=index,columns=columns)
            print raw_data.shape                                                       
            #VAR Model
            model=VAR(raw_data)            
            result=model.fit(3,trend="nc")
            print result.k_ar
            coefs=result.coefs
            #result.plot()       
            var_parameters.append(self.getInformation(coefs))       
            #end for
        var_parameters=np.asarray(var_parameters)    
        print "var_parameters is in shape (%s,%s)" %(var_parameters.shape[0],var_parameters.shape[1])
        return var_parameters
    
    def getInformation(self,coefs):
        '''
        Get Information from the Time Series Model
        Input: coefficients of VAR/AR model
        Output: var_parameter for one patient 1d-array
        '''
        #flatten the coefs
        var_parameter=coefs.flatten()
        #convert coefs to an array of matrices, of which form it originally is
        """
        coefs_matrix=[]
        for coef in coefs:
            matrix=np.matrix(coef)
            coefs_matrix.append(matrix)
        coefs_matrix=np.asarray(coefs_matrix)
        #take eigenvalues of coef
        eigenvalue_list=[]
        for matrix in coefs_matrix:
            eigenvalues,eigenvectors=np.linalg.eig(matrix)
            eigenvalue_list.append(eigenvalues)
        var_parameter=np.asarray(eigenvalue_list).flatten()
        """
        return var_parameter
        
    def sample_model(self):
        """
        Consider each time stamp of graph is a sample from a generative network model
        """
        var_parameters=[]
        for file_name,g in self.data:
            raw_data=self.get_graph_measures(g)
            #average graph measures over time series(point estimator to the graph measure)
            est_graph_measure=raw_data.mean(axis=0)
            var_parameters.append(est_graph_measure)
        var_parameters=np.asarray(var_parameters)
        return var_parameters
        
    
    def build_AR_model(self):
        var_parameters=[]
        print self.data[1]
        for file_name,g in self.data:    
            raw_data=self.get_graph_measures(g)
            assert raw_data.shape[1]==1,"AR can not be used, please use VAR model"

            #######################################################################filter in time series model
            #convert to array-list input with shape as (N,)            
            original_raw_data=[x for nested in raw_data for x in nested]
            #Filter the noise in raw_data
            raw_data=savitzky_golay(original_raw_data, window_size=31, order=12)
            #plot the filtering effect
            """
            index=range(len(original_raw_data))
            #print raw_data
            plt.plot(index, original_raw_data, label='Noisy signal')
            plt.plot(index, raw_data, 'r', label='Filtered signal')
            plt.legend()
            plt.show()
            """
            ################################################################################
            
            columns=vars.graph_measurenames
            #prepare input data for VAR model
            #time series model put stricts on index, it must exist and has to be DatatimeIndex
            index=pd.date_range("1",periods=len(raw_data))
            raw_data = pd.DataFrame(raw_data,index=index,columns=columns)
            #print raw_data.shape
     
            """
            choose best lag
            a=zip(*(raw_data.to_records(index=False).tolist()))
            a=[ _ for _ in a[0]] 
                            
            x=sts.acf(a)
            
            plt.plot(range(len(x)),x,label="Autocorrelation")
            plt.legend()
            plt.show()
            """
                
                
            #AR Model
            model=AR(raw_data)
            #best_lag=model.select_order(12,ic="aic")
            #print "best_lag of AR: %f" %best_lag
            result=model.fit(1)
            print result.params
            coefs=np.asarray(result.params)
            #plt.plot(range(len(raw_data)),raw_data)

            #Get Information from the Time Series Model        
            flatten=coefs.flatten()
            var_parameters.append(flatten)       
            #end for
        var_parameters=np.asarray(var_parameters)    
        print "var_parameters is in shape (%s,%s)" %(var_parameters.shape[0],var_parameters.shape[1])
        return var_parameters
        
        
    def clusteringTimeSeries(self):
        """
        Aggregate similar graph along the time series
        Start from t=0, start a cluster, scanning along the time series axis, if the distance of the graph between t=i and t=i+1
        is small, the we aggregate these two graphs to this cluster. If the distance between two graphs is large, then we start a new
        cluster. So only the adjacent graph has a possibility to be aggregated.
        """
        
        pass

"""
#plot the fitted values on original data
fittedvalues=result.fittedvalues
#print for each col
try:
    for col in columns:
        x1=range(len(raw_data))
        delta_x=len(raw_data)-len(fittedvalues) #this should equal result.k_ar
        x2=range(delta_x+1,len(x1)+1)
        plt.figure()
        if tsa=="AR":
            plt.plot(x1,raw_data[col],x2,fittedvalues,'k--') 
        elif tsa=="VAR":
            plt.plot(x1,raw_data[col],x2,fittedvalues[col],'k--')
        else:
            raise ValueError()
except ValueError:
    print "can not plot data from this type of model"
""" 


        
if __name__=="__main__":
    """
    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    #selector=Selector("all",weight_filter_method="discretize")
    data=selector.data
    model=TSAModel(data,vars.tsa_model_name)
    var_parameters=model.build_model()
    print var_parameters.shape
    
    Y=selector.build_clinical_variables()
    Y=[y[0] for y in Y]
    print Y
    #assert var_parameters.shape[1]==1,"the shape of X is wrong"
    X1=[x[0] for x in var_parameters]
    X2=[x[1] for x in var_parameters]
    print X1
    print X2
    fig=pl.figure()
    pl.scatter(X1,X2,c=Y,s=80,cmap=pl.cm.Paired)
    pl.title("HRP_DA")
    pl.xlabel("K")
    pl.ylabel("rewiring probability")
    pl.xlim(min(X1),max(X1))
    pl.ylim(min(X2),max(X2))
    
    fig = pl.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    #X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X1, X2,Y,s=80,cmap=pl.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("K")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("rewiring probability")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("clinical variable")
    ax.w_zaxis.set_ticklabels([])           
    
    pl.show()
    """
    training_data=[1,2,3,1,2,4,1,2,3,2,3,1,2]
    
    