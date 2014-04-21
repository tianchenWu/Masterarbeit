# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 19:14:38 2014

@author: wu
"""
import numpy as np
import math
from test.I_O_put.Selector import *
import test.Global.globalvars as vars
import Pycluster as pc
import matplotlib.pylab as pl

class MarkovModelBuilder:
    """
    Initialize and train a Markov Model
    """
    def __init__(self,states):
        """
        Input:
        """
        self.states=states
        self.num_states=len(self.states)
        #a dictionary{numerical value of state:index of state}, index of state starts from 0
        self.statesDict={}
        for index,state in enumerate(self.states):
            self.statesDict[state]=index
        #row sum up to one
        self.freq_matrix=np.zeros((self.num_states,self.num_states))
        #self.trans_matrix=np.zeros((self.num_states,self.num_states))
    
    def assignState(self,e):
        """
        Given element of sequence, find state that generate this element
        For rough set implementation the generative state is deterministic
        Input:numerical number
        Output: state index
        """
        state=min(self.states, key=lambda x:abs(x-e))
        return self.statesDict[state]
        
    def fit(self,sequence):
        """
        Learn a Markov Model based on the given sequence
        Input: sequence , a list
        """
        #updata frequency matrix
        for i in range(len(sequence)-1):
            _from=self.assignState(sequence[i])
            _to=self.assignState(sequence[i+1])
            self.freq_matrix[_from][_to]=self.freq_matrix[_from][_to]+1
        #print "Frequency Matrix:"
        #print self.freq_matrix
        #convert frequency matrix to probability matrix, namely transition matrix
        self.trans_matrix=self.freq_matrix.copy()
        for i in range(self.num_states):
            freq_sum=0
            for j in range(self.num_states):
                freq_sum=freq_sum+self.freq_matrix[i][j]
            if freq_sum==0:
                pass
            else:            
                for j in range(self.num_states):
                    self.trans_matrix[i][j]=float(self.freq_matrix[i][j])/float(freq_sum)
        #print "Transition Matrix:"
        #print self.trans_matrix
        return self
    
    def calProb(self,sequence):
        """
        Estimate the log probability(base e) of the sequence given model
        """
        assert self.trans_matrix is not None,"First train the model!!"
        MIN_VAL=np.min(self.trans_matrix[self.trans_matrix>0])
        accum_log_prob=0
        for i in range(len(sequence)-1):
            _from=self.assignState(sequence[i])
            _to=self.assignState(sequence[i+1])
            prob=self.trans_matrix[_from][_to]
            if prob==0:
                prob=MIN_VAL#assign the minimal value in the whole transition matrix
            accum_prob=float(accum_log_prob)+float(math.log(prob))
        return accum_prob


###################################################################################
states=np.linspace(0.05,0.95,19)#len 19
print states
#get one dimensional time series data for all patients,if graph measure is one dimensional
def get_seqs(data):
    """
    Input:data
    Output: a list of patient name, a list of measures
    """
    name_rooster=[]
    seq_list=[]
    for file_name,g in selector.data:
        name_rooster.append(file_name)
        seq_list.append([measure[0] for measure in g.measures])#only make full sense for 1d data
        #print seq_list
    return name_rooster,seq_list
    


def construct_dist_matrix(seq_list):
    """
    build pairwise conditionally probability matrix for all patients, condition on the 
    model build for one patient, calculate the probability,other patients coming from this model
    construct distance matrix fullfilled:
    1. symetric
    2. distance to self is zero
    
    Input: pairwise path probability
    Output: distance matrix
    """
    num_seq=len(seq_list)
    pairwise_path_prob_matrix=np.zeros((num_seq,num_seq))
    for i in range(num_seq):
        #build mm model for i th sequence/patient
        i_mm=MarkovModelBuilder(states).fit(seq_list[i])
        for j in range(num_seq):
            if i==j:
                continue #the probability of i th sequence generated from i_mm is not 0, but for the sake of saving computation
            j_prob=i_mm.calProb(seq_list[j])
            pairwise_path_prob_matrix[i][j]=-j_prob#take negative make distance
    dist_matrix=pairwise_path_prob_matrix.copy()
    #make it symmetric
    for i in range(num_seq):
        for j in range(i+1,num_seq):
            #weighted average of item (i,j) and (j,i)
            #weight is calculated by the length of sequence,more long the sequence, the model based on it
            #is assumed to be more reliable
            ij=dist_matrix[i][j]
            len_ij=len(seq_list[i])
            ji=dist_matrix[j][i]
            len_ji=len(seq_list[j])
            avg=float(len_ij*ij+len_ji*ji)/float(len_ij+len_ji)
            dist_matrix[i][j]=dist_matrix[j][i]=avg
    return dist_matrix




if __name__=="__main__":
    """
    training_data=[1,2,3,1,2,4,1,2,3,2,3,1,2,4,3,2]
    test_data=[1,2,3,1,2,3,2]
    mm=MarkovModelBuilder([1,2,3,4]).fit(training_data)
    prob=mm.calProb(test_data)
    print prob
    """
    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    data=selector.data
    cv=selector.build_clinical_variables()
    cv_out=[e[0] for e in cv]
    print "cv: %s" %cv_out
    
    """
    name_rooster,seq_list=get_seqs(data)
    dist_matrix=construct_dist_matrix(seq_list)
    #print "dist_matrix:"
    #print dist_matrix
    
    #cluster patients based on dist_matrix
    clusterid, error, nfound = pc.kmedoids (dist_matrix, nclusters=3, npass=100,initialid=None)
    fig,ax=pl.subplots(1)
    print "clusterid: %s" %clusterid
    print "error: %s" %error
    print "nfound: %s" %nfound
    ax.scatter(clusterid,cv)
    """


    