# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:19:51 2014

@author: wu
"""

from Models.TSAModel import TSAModel
import numpy as np
from Models.MLModel import RegressionModel as RM
from I_O_put.Selector import Selector
from Global.HelpFunction import nan_filter,normalizer,dr
import Global.globalvars as vars
import sklearn.cross_validation as CV
from scipy.stats import sem
import matplotlib.pylab as pl
from scipy.stats import norm
import matplotlib.mlab as mlab

class Evaluation:
    def __init__(self,selector):
        self.data=self.build_graph_model(selector)
        self.var_parameters=self.build_tsa_model(self.data)
        
        ########################################################preprocessing for machine learning model
        #dimensionality reduction
        #self.var_parameters=dr(self.var_parameters,dim="mle")
        self.var_parameters=dr(self.var_parameters,dim=20)
        #print "var_parameters after PCA %s" %self.var_parameters
        self.clinical_variables=selector.build_clinical_variables()
        #filter/handle nan value in clinical_variables, corresponding data in var_parameters
        self.var_parameters,self.clinical_variables=nan_filter(self.var_parameters,self.clinical_variables)
        #normalize var_parameters and clinical_variables
        self.var_parameters=normalizer(self.var_parameters)
        self.var_parameters,self.clinical_variables=np.asarray(self.var_parameters),np.asarray(self.clinical_variables)
        print type(self.var_parameters),type(self.clinical_variables)        
        assert len(self.var_parameters)==len(self.clinical_variables),"length of regressors and independent variables mismatch"
        ###########################################################
    
    def build_graph_model(self,selector):
        """
        Input: selector object, selector.data contains the graph model we have built
        in the form of (file_name,EEGGraph)
        
        Output:[(file_name,EEGGraph),....] 
        """
        return selector.data

    def build_tsa_model(self,data):
        """
        Input: output from build_graph_model
        Output: var_parameters [2d array][(n_samples,n_features)]
        """
        return TSAModel(self.data,vars.tsa_model_name).build_model()
        
  
    def build_ml_model(self,var_parameters,clinical_variables):
        """
        Input: X(training data set),y(corresponding targets)
        Output:the desired machine learning model
        """
        return RM(var_parameters,clinical_variables)
        

    
    def evaluate(self):
        X=self.var_parameters
        Y=self.clinical_variables
        errors=[]
        K=15
        for i in range(10):
            #loo = CV.LeaveOneOut(len(y))
            #lpo = CV.LeavePOut(len(Y), 2)
            kf = CV.KFold(len(Y), n_folds=K, indices=False,shuffle=True)
            
            #scores=[]
            error=[]#mean square error
            for train, test in kf:
                #print("%s %s" % (train, test))
                #analyse_multicolinearity(X[train])
                model=self.build_ml_model(X[train],Y[train]).SLR()
                #score=model.score(X[test],Y[test])
                #score=model.score(X[train],Y[train])
                #scores.append(score)
                #print "shape of training data (%s,%s)" %(X[train].shape[0],X[train].shape[1])
                predicted=self.model_prediction(model, X[test])
                z=[Y[test][i]-predicted[i] for i in range(len(predicted))]                        
                error.extend(z)
            errors.append(np.mean(np.asarray([np.abs(x[0]) for x in error])))
        
        #print self.mean_score(scores)
        #self.analyze_eval_result(errors)
        print errors
        return np.asarray(errors)
        #plot_predict_error(errors)
        
    def model_prediction(self,model,X_test):
        """
        Input: test data
        Output: a list of errors
        """
        X_test=np.asarray(X_test) 
        assert X_test.shape,"invalid form of input"
        #print "shape of validation data (%s,%s)" %(X_test.shape[0],X_test.shape[1])        
        return model.predict(X_test)
    
    def analyze_eval_result(self,errors):
        """
        Input should be in form of [array([-0.20340336]), array([-0.12608339]),...]
        """
        #convert the input into 1d array([x1,x2,x3,...]),take the absolute value
        #errors=np.asarray([np.abs(x) for x in errors])
        errors=np.asarray(errors)
        mean=errors.mean(axis=0)
        std=errors.std(axis=0)
        print "absolute mean:  %s, standard deviation:  %s" %(mean,std)
        return mean,std
    
        
    def mean_score(self,scores):
        """Print the empirical mean score and standard error of the mean"""
        print scores
        print "min: %s" % np.min(scores)
        print "len: %s" % len(scores)
        return "Mean score: {0:.3f} (+/- {1:.3f})".format(np.mean(scores),sem(scores))
    #def plot_score(self,scores):
    #    _=pl.hist(scores,bins=30,alpha=0.2)
        
        
        
        
        
if __name__=="__main__":
    selector=Selector("20",weight_filter_method=vars.weight_filter_method)
    #selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    eval=Evaluation(selector)
    eval.evaluate()
