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
#print healthy_data
healthy_X=TSAModel(healthy_data,"VAR").build_model()
disease_X=TSAModel(disease_data,"VAR").build_model()
X=np.concatenate((healthy_X,disease_X),axis=0)

y=np.asarray(["h"]*15+["d"]*15)

print len(X),len(y)


#support vector machine classification
from sklearn import svm,neighbors
from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold,Bootstrap
from sklearn.grid_search import GridSearchCV
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB


avg_accuracy=[]
N=1000
K=15
bs=Bootstrap(len(y),n_iter=N,random_state=0)
for train_index, test_index in bs:
    #innerloop is leave one out
    index=np.concatenate((train_index,test_index),axis=0)
    count_=0
    sum_=0
    loo = LeaveOneOut(len(y))
    kf = KFold(len(y), n_folds=K, indices=False,shuffle=True)
    """
    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=kf)
    grid.fit(X, y)
    print("The best classifier is: ", grid.best_estimator_)
    """
    for train, test in loo:    
        train=index[train]
        test=index[test]
        train=list(set(train) - set(test))#get the test data out of train data
        assert len(test)==1,"error in loo"
        #clf = svm.SVC(C=1000,gamma=8)
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights="distance")
        #clf = LDA()
        #clf= GaussianNB()
        clf.fit(X[train], y[train])
        #print clf
        predicted=clf.predict(X[test])
        tested=y[test]
        #print predicted,tested
        correct=np.sum(np.asarray([1 for i in range(len(y[test])) if predicted[i]==tested[i]]))
        #print correct
        count_=count_+correct
        sum_=sum_+len(predicted)
    accuracy=float(count_)/float(sum_)
    print accuracy
    avg_accuracy.append(accuracy)
    
avg_accuracy=np.asarray(avg_accuracy)

print "mean: %s , std: %s" % (np.mean(avg_accuracy), np.std(avg_accuracy))

