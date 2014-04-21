# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:18:47 2014

@author: wu
"""

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import statsmodels.api as smapi


class RegressionModel():
    def __init__(self,var_parameters,clinical_variables):
        self.var_parameters=var_parameters
        self.clinical_variables=clinical_variables
        assert len(self.var_parameters)==len(self.clinical_variables),"the length of regressors and dependent variables should be the same"
    
    """
    #self.preprocessing()
    def preprocessing(self):
        
        thow away "nan" data, from clinical_variables and corresponding var_parameters
        
        reserved_data=[(i,_) for (i,_) in enumerate(self.clinical_variables) if not np.isnan(_).any()]
        index,self.clinical_variables=zip(*reserved_data)
        self.clinical_variables=np.asarray(self.clinical_variables)
        self.var_parameters=np.asarray([self.var_parameters[i] for i in index])
    """
    def SLR(self):
        """
        Simple Linear Regression
        return: regression model
        """
        
        """
        #var_parameters against clinical_variables
        anova_filter = SelectKBest(f_regression, k=20)
        #print "Scores of features: %s" %anova_filter.scores
        clf = linear_model.LinearRegression()
        anova_slr = Pipeline([('anova', anova_filter), ('slr', clf)])
        anova_slr.fit (self.var_parameters,self.clinical_variables)
        #print clf.coef_
        """
        
        clf = linear_model.LinearRegression()
        clf.fit(self.var_parameters,self.clinical_variables)
        """
        clf = smapi.OLS(self.var_parameters, self.clinical_variables).fit()
        # Find outliers #
        test = clf.outlier_test()
        print test
        """
        return clf
        
    def RR(self,alpha):
        """
        Ridge Regression
        return: regression model
        """
        clf = linear_model.Ridge (alpha)
        clf.fit (self.var_parameters,self.clinical_variables)
        return clf
    def Lasso(self,alpha):
        clf = linear_model.Lasso(alpha)
        clf.fit (self.var_parameters,self.clinical_variables)
        return clf
        
        
        
        
        