# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:42:35 2014

@author: wu
"""

import numpy as np
import pandas as pd
from test.I_O_put.Selector import *
from sklearn.decomposition import PCA
import igraph as ig
from test.Models.GraphModel import EEGGraph
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import test.Global.globalvars as vars
from test.Global.HelpFunction import avg_wpl,avg_wcc
from test.StatsAnalysis.NetworkSimulation import simulateWeightedRandomGraph

def estGraphMeasureTable(EEGGraph):
    """
    Establish the graph measure table
    Input: time series EEG of one patient, igraph object
    Output: graph measure table of one patient, dataFrame
    """
    table=NA(EEGGraph).extractInfo().dropna()
    return table

def get_edge_number(graph):
    if not graph.is_weighted():
        count=graph.ecount()
    else:
        count=sum(graph.es["weight"])
        
    return count


def get_average_path_length(graph):
    """
    average_path_length
    """    
    if not graph.is_weighted():
        count=graph.average_path_length(directed=False, unconn=False)
    if graph.is_weighted():
        #print "call weighted version average path length"
        g=graph.copy()
        #nomalize graph weight
        max=np.max(np.array(graph.es["weight"]))
        g.es["weight"]=g.es["weight"]/max
        #calculated weigted avg path length on normalized graph
        count=avg_wpl(g)
        
    return count

def get_clustering_coefficient(graph):
    """
    clustering coefficient
    """
    if not graph.is_weighted():
        count=graph.transitivity_avglocal_undirected(mode="zero")
    if graph.is_weighted():
        #print "call weighted version clustering coeffcient"
        count=avg_wcc(graph,weights="weight")
    
    return count
    
def get_motif3(graph):
    count=graph.motifs_randesu_no(size=3)
    return count
def get_motif4(graph):
    count=graph.motifs_randesu_no(size=4)
    return count
    
def get_numClique(graph):
    count=graph.clique_number()
    return count
    
def get_hub_score(graph):
    """
    Kleinberg's hub score
    """
    if not graph.is_weighted():
        count=np.mean(np.asarray(graph.hub_score()))
    else:        
        count=np.mean(np.asarray(graph.hub_score(weights="weight")))
    return count
    
def get_max_hub_score(graph):
    
    if not graph.is_weighted():
        count=np.max(np.asarray(graph.hub_score()))
    else:        
        count=np.max(np.asarray(graph.hub_score(weights="weight")))
    return count    
    
    
class NA():
    def __init__(self,eeg_graph):
        #if the input is an object of EEG
        if isinstance(eeg_graph,EEGGraph):
            self.graphlist=eeg_graph.graphlist
        #this is an object of Graph
        else:
            self.graphlist=[eeg_graph]
        self.measure_names=["numclique","transitivity_local","transitivity_global","density","diameter","girth","radius","numindependence","motif3","motif4","assortativity","apl","max_betweenness_centrality","max_eigenvector_centrality"]
        
    
    #extract measure informations from graphlist reuten a dataFrame(table)
    #which dates is the order of the graph in graphlist
    #columns are the considered graph measures(global):
    #assortativity
    #(weighted) average path length,unconn=False
    #clique number
    #transitivity(global,local) mode=zero
    #density
    #diameter
    #girth
    #radius
    #independence_number
    #number of motifs of sizes 3 and4
    
    def extractInfo(self):
        m=len(self.measure_names)
        n=len(self.graphlist)
        temp=np.zeros((n,m))
        for (c1,graph) in enumerate(self.graphlist):
            for (c2,measure) in enumerate(self.measure_names):
                if measure=="assortativity":
                    temp[c1,c2]=graph.assortativity_degree(directed=False)
                if measure=="apl":
                    if not graph.is_weighted():
                        temp[c1,c2]=graph.average_path_length(directed=False, unconn=False)
                    if graph.is_weighted():
                        g=graph.copy()
                        #nomalize graph weight
                        max=np.max(np.array(graph.es["weight"]))
                        g.es["weight"]=g.es["weight"]/max
                        print g.es["weight"]
                        #calculated weigted avg path length on normalized graph
                        temp[c1,c2]=avg_wpl(g)
                if measure=="numclique":
                    temp[c1,c2]=graph.clique_number()
                if measure=="transitivity_local":
                    if not graph.is_weighted():
                        temp[c1,c2]=graph.transitivity_avglocal_undirected(mode="zero")
                    if graph.is_weighted():
                        temp[c1,c2]=np.mean(graph.transitivity_local_undirected(vertices=graph.vs, mode="zero", weights="weight"))
                if measure=="transitivity_global":
                    temp[c1,c2]=graph.transitivity_undirected(mode="zero")
                if measure=="density":
                    temp[c1,c2]=graph.density()
                if measure=="diameter":
                    temp[c1,c2]=graph.diameter(directed=False, unconn=True)
                if measure=="girth":
                    temp[c1,c2]=graph.girth()
                if measure=="radius":
                    temp[c1,c2]=graph.radius()
                if measure=="independence_number":
                    temp[c1,c2]=graph.independence_number()
                if measure=="motif3":
                    temp[c1,c2]==graph.motifs_randesu(size=3)
                if measure=="motif4":
                    temp[c1,c2]==graph.motifs_randesu(size=4)
                if measure=="max_betweenness_centrality":
                    if not graph.is_weighted():
                        temp[c1,c2]==np.max(graph.betweenness(vertices=graph.vs, directed=False, nobigint=False))
                    if graph.is_weighted():
                        temp[c1,c2]==np.max(graph.betweenness(vertices=graph.vs, directed=False,weights="weight", nobigint=False))
                if measure=="max_eigenvector_centrality":
                    if not graph.is_weighted():
                        temp[c1,c2]==np.max(graph.eigenvector_centrality(directed=False,weights=None))
                    if graph.is_weighted():
                        temp[c1,c2]==np.max(graph.eigenvector_centrality(directed=False,weights="weight"))
        #convert to dataFrame
        table=pd.DataFrame(data=temp,columns=self.measure_names)
        #droped=table.dropna()
        #print "--------------------------------"
        #print droped
        return table
    
    #run after extractInfo
    
    def saveToCSV(self,table):
        path="E:\\Program Files\\PyderWorkspace\\ma\\Graph_result\\hi.txt"
        cols=self.measure_names
        table.to_csv(path,cols=cols)

if __name__=="__main__":
    ############################set up data
    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    #selector=Selector("5",weight_filter_method=weight_filter_method)
    data=selector.data
    #table_list=[]
    X=[]
    for (file_name,g) in data:
        networkAnalyzer=NA(g)
        #graph measures for one patient
        table=networkAnalyzer.extractInfo()
        #filter the data
        table=table.dropna()
        x=table.mean()
        #convert from series to list
        x=[value for (index,value) in x.iteritems()]
        X.append(x)
        print x
        #table_list.append(table)
        #networkAnalyzer.saveToCSV(table)
        

        
    #comtable=pd.concat(table_list)
    #save this table to text
    #filter the nan valued caused by assortativity calculation
    #comtable=comtable.dropna(axis=0)
    #print comtable
    ###############################################################
    ######################################################visualize data
    X=np.asarray(X)
    y=selector.build_clinical_variables()#it is now a ndarray in shape (n_samples,)
    #convert y into 1 dim
    y=[x for list in y for x in list]
    print y
    
    
    """
    #target on clinical variable defined in globalvars
    #dimensionality reduction
    from sklearn.decomposition import RandomizedPCA,PCA
    
    #Three dimensional
    fig = pl.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],s=80, c=y,
           cmap=pl.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])           
    
    pl.show()
    """
        