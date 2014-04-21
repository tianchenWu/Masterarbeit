# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:52:41 2014

@author: wu
"""

#split the regressor_list into training data and test data, if the result has residual, then there must be one group has less
    #members than the others
    def k_fold_cross_validation(self,K, randomise = False):
        """
        Generates K (training, validation) pairs from the items in X.
    
        Each pair is a partition of X, where validation is an iterable
        of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
    
        If randomise is true, a copy of X is shuffled before partitioning,
        otherwise its order is preserved in training and validation.
        """
        #X is a list of tuples with two components, each of which is an array
        X=zip(self.var_parameters,self.clinical_variables)
        if randomise: from random import shuffle; X=list(X); shuffle(X)
        for k in xrange(K):
            training = [x for i, x in enumerate(X) if i % K != k]
            validation = [x for i, x in enumerate(X) if i % K == k]
            #for x in X: assert (x in training) ^ (x in validation), "cross_validation error"
            yield training, validation
            
    def leave_one_out(self,randomise=False):
        '''
        call k_fold_cross_validation, K=number of data after filtering nan
        '''
        for training,validation in self.k_fold_cross_validation(len(self.var_parameters),randomise=randomise):
            yield training,validation
    
    #return: predicted values , shape = (n_samples,),1d-array
    def model_prediction(self,model,v_var_parameters):
        v_var_parameters=np.asarray(v_var_parameters)
        assert v_var_parameters.shape,"invalid form of input"
        print "shape of validation data (%s,%s)" %(v_var_parameters.shape[0],v_var_parameters.shape[1])        
        return model.predict(v_var_parameters)
        
    
    def analyze_eval_result(self,errors):
        """
        Input should be in form of [array([-0.20340336]), array([-0.12608339]),...]
        """
        #convert the input into 1d array([x1,x2,x3,...]),take the absolute value
        errors=np.asarray([np.abs(x[0]) for x in errors])
        mean=errors.mean(axis=0)
        std=errors.std(axis=0)
        #print "mean:  %s, standard deviation:  %s" %(mean,std)
        return mean,std
        
        
    def evaluate(self):
        #list of id-arrays with only one element
        #[array([-0.20340336]), array([-0.12608339]), array([-0.03167718]), array([-0.05273624]), array([ 0.27342744])]
        errors=[]
        # training/validation data in form of [(array of var_para,array of cvs),...]
        for training,validation in self.leave_one_out():
            assert not len(validation)==0,"the validation list should not be empty"
            #list of arrays
            t_var_parameters=[var_parameters for (var_parameters,clinical_variables) in training]
            t_clinical_variables=[clinical_variables for (var_parameters,clinical_variables) in training]
            model=self.build_ml_model(t_var_parameters,t_clinical_variables)
            #predict using the model and  given validation regressors
            v_var_parameters=[var_parameters for (var_parameters,clinical_variables) in validation]
            v_clinical_variables=[clinical_variables for (var_parameters,clinical_variables) in validation]
            for i in v_clinical_variables:assert not np.isnan(i).any(),"nan value v_clinical_variables"
            v_predicted_clinical_variables=self.model_prediction(model, v_var_parameters)
            #compare fitted values and observed values
            
            z=[v_clinical_variables[i]-v_predicted_clinical_variables[i] for i in range(len(v_clinical_variables))]                        
            errors.extend(z)            
            #print len(training)
            #print len(validation)
        self.analyze_eval_result(errors)
        return errors


def find_all_paths(graph, start, end, path=[]):
    """
    return all possible paths between two node start and end
    if no path is available return empty list
    Output:a list of path(each of which is a list of nodes)
    """
    def find_all_paths_aux(adjlist, start, end, path):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in adjlist[start] - set(path):
            paths.extend(find_all_paths_aux(adjlist, node, end, path))
        return paths

    adjlist = [set(graph.neighbors(node)) for node in xrange(graph.vcount())]
    return find_all_paths_aux(adjlist, start, end, [])
    
def calculateMaximalWeightedPathLength(graph,paths):
    """
    If there is no connection between end and start
    F value will be set to 1/N
    Output: weighted path length for each path from paths in a list
    """
    MIN_F=float(1)/float(vars.N)#min F means maximal path length
    w_list=[]
    if paths==[]:
        #no path between start and end,this point is not defined
        #print "paths empty"
        F=MIN_F
        F=float(1)/float(F)
        return F
    for path in paths:
        w=1
        assert len(path)>=2,"something wrong in path calculation"
        for i in range(len(path)-1):
            eid=graph.get_eid(path[i], path[i+1], directed=False, error=True)
            e=graph.es(eid)
            #print e
            w_i=e["weight"][0]
            #print w_i
            w=w*w_i
            #print w
        w_list.append(w)
    #print w_list
    F=np.max(w_list)
    if F<MIN_F:
        F=MIN_F
    assert F>=MIN_F,"The maximal path length should not be larger than N"
    F=float(1)/float(F)
    #print F
    return F
        
    
def weightedPathLength(graph,v):
    """
    Weighted path length for one vertice i in the graph
    calculating based on formula from Marcos Bolanos in paper 
    "A weighted small world network measure for assessing functional connectivity"
    
    Output: path length
    """
    J=set([s.index for s in graph.vs])-set([graph.vs[v].index])
    answer=0
    for j in J:
        #print "vertex: %s" %j
        sum_F=calculateMaximalWeightedPathLength(graph,find_all_paths(graph,v,j))
        answer=answer+sum_F
    answer=float(answer)/float(vars.N-1)
    print "answer: %s" %answer
    return answer
        
        
if __name__=="__main__":
    #selector=Selector("20",weight_filter_method=weight_filter_method)
    selector=Selector("all",weight_filter_method=weight_filter_method)
    eval=Evaluation(selector)
    #errors=eval.evaluate(numfold=2)
    errors=eval.evaluate()
    print "result----------------------------"
    print errors
    mean,std=eval.analyze_eval_result(errors)
    print "mean:  %s, standard deviation:  %s" %(mean
    


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
        matrix=np.asarray(g.get_adjacency().data)#type list of list-> numpy array
        print type(matrix)
        np.save(patientStampPath,matrix)
        #g.write_pajek(patientStampPath)
        
        break
    break
    #g=average_graph(eeg.graphlist,0.5)
    #print g.vs["name"]
    #print ig.Graph.degree(g.vs)
    #g.write(patientPath,format="pajek")
    #print g
    #break
    """
    for g in eeg.graphlist:
        print g
        patientPath=patientsNetworkDir+"\\"+parse_file_name(file_name)+"right.net"
        g.write_pajek(patientPath)
        #layout=g.layout("kk")
        #ig.plot(g,layout=layout)
        break
    """



"""
#write out
selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data

healthy_data,disease_data=data
patientsNetworkDir="C:\\Users\\wu\\Desktop\\Patient"

#healthy group
for file_name,eeg in healthy_data:
        
    graph=average_graph(eeg.graphlist,0.5)
    
    patientPath=patientsNetworkDir+"\\"+"Healthy_"+parse_file_name(file_name)+".gml"
    print patientPath
    graphPath=patientPath+".pajek"
    graph.write_gml(graphPath)
    #layout=graph.layout("kk")
    #ig.plot(graph,layout=layout)   
"""
"""
######################################################
#this should be used

#eye disease group
for file_name,eeg in disease_data:
    #graph=average_graph(eeg.graphlist,0.5)
    patientPath=patientsNetworkDir+"\\"+"Disease\\"+parse_file_name(file_name).split("_")[1]
    print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".net"
        g.write_pajek(patientStampPath)
    #graphPath=patientPath+".net"
    #graph.write_pajek(graphPath)

#########################################################
#read in
"""

#a=ig.Graph.Read_Pajek("C:\\Users\\wu\\Desktop\\Patient\\alpha_AM_prae_closed1.pajek")
#layout=a.layout("kk")
#ig.plot(a,layout=layout)


    """
    b1=pl.boxplot(errors,0,'')
    pl.setp(b1['boxes'], color='black')
    pl.setp(b1['whiskers'], color='black')
    pl.setp(b1['medians'], color='black')
    pl.setp(b1['caps'], color='black')
    """
