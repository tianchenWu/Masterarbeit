# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 19:30:26 2014

@author: wu
"""
from os import listdir
import numpy as np
from statsmodels.tsa.vector_ar.var_model import *
from test.Filter.LowPassFilter import savitzky_golay

#read in data
root="C:\Users\wu\spyder_workspace\HurryUp2\TempData\Binary ERGM Classification"

healthy_patients_path="C:\\Users\\wu\\spyder_workspace\\HurryUp2\\TempData\\Binary ERGM Classification\\Healthy"
healthy_patients=[]

for file in listdir(healthy_patients_path):
    file_fullpath=healthy_patients_path+"\\"+file
    f=open(file_fullpath)
    patient=[]
    for line in f.readlines():
        density,star,triangle=line.split(" ")
        triangle,_=triangle.split("\n")
        density,star,triangle=float(density),float(star),float(triangle)
        if np.isinf(star): star=1
        if np.isinf(density): density=1
        if np.isinf(triangle): triangle=1
        print density,star,triangle
        patient.append((density,star,triangle))
    patient=np.asarray(patient)
    healthy_patients.append(patient)
    
disease_patients_path="C:\\Users\\wu\\spyder_workspace\\HurryUp2\\TempData\\Binary ERGM Classification\\Disease"
disease_patients=[]

for file in listdir(disease_patients_path):
    file_fullpath=disease_patients_path+"\\"+file
    f=open(file_fullpath)
    patient=[]
    for line in f.readlines():
        density,star,triangle=line.split(" ")
        triangle,_=triangle.split("\n")
        density,star,triangle=float(density),float(star),float(triangle)
        patient.append((density,star,triangle))
    patient=np.asarray(patient)
    disease_patients.append(patient)
    
patients=np.concatenate((healthy_patients,disease_patients),axis=0)
y=["h"]*15+["d"]*15
X=[]
#########################################################################
#processing time series
#VAR model

for patient in patients:
    
    ###############################################################################filter in time series model
    #group the data by measure
    original_raw_data=patient.swapaxes(0,1);
    raw_data=[]
    #the form of data_each_measure is 1d numpy array
    for data_each_measure in original_raw_data:
        filtered_data_each_measure=savitzky_golay(data_each_measure, window_size=31, order=12)
        raw_data.append(filtered_data_each_measure)
    patient=np.asarray(raw_data).swapaxes(1,0)
    #print raw_data
    ################################################################################
    
    model=VAR(patient)
    result=model.fit(3,trend="nc")
    coefs=(result.coefs).flatten()
    X.append(coefs)

X=np.asarray(X)
y=np.asarray(y)

#print X
#######################################################################
#Classification

from sklearn import svm,neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold,Bootstrap
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB


#loo = LeaveOneOut(len(y))
#lplo = LeavePLabelOut(labels, 2)
#ss = ShuffleSplit(5, n_iter=3, test_size=0.25,random_state=0)

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
        #assert len(test)==1,"error in loo"
        #clf = svm.SVC(C=10,gamma=0.1)
        #clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights="distance")
        #clf = LDA()
        clf= GaussianNB()
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


























  
