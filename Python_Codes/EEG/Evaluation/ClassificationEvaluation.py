# -*- coding: utf-8 -*-
"""
Created on Sun Mar 09 15:22:43 2014

@author: wu
"""
from test.Models.TSAModel import TSAModel
import numpy as np
from test.I_O_put.Selector import Selector
import test.Global.globalvars as vars



selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
healthy_data,disease_data=selector.data
print healthy_data
healthy_X=TSAModel(healthy_data,"AR").build_model()
disease_X=TSAModel(disease_data,"AR").build_model()
X=np.concatenate((healthy_X,disease_X),axis=0)

y=np.asarray(["h"]*15+["d"]*15)

print len(X),len(y)


from sklearn import svm,neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold

kf = KFold(len(y), n_folds=15, indices=False)
loo = LeaveOneOut(len(y))
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=loo)
grid.fit(X, y)
print("The best classifier is: ", grid.best_estimator_)

count=0
for train, test in loo:    
    
    clf = svm.SVC(C=10.0,gamma=10)
    #clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights="distance")
    clf.fit(X[train], y[train])
    #print clf
    print clf.predict(X[test]),y[test]
    if clf.predict(X[test])==y[test]:
        count=count+1
    

print count
