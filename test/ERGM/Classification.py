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
root="C:\\Users\\wu\\Desktop\\TimeSeriesCoefficient"

healthy_patients_path="C:\\Users\\wu\\Desktop\\TimeSeriesCoefficient\\Healthy"
healthy_patients=[]

for file in listdir(healthy_patients_path):
    file_fullpath=healthy_patients_path+"\\"+file
    f=open(file_fullpath)
    patient=[]
    for line in f.readlines():
        density,star,triangle=line.split(" ")
        triangle,_=triangle.split("\n")
        density,star,triangle=float(density),float(star),float(triangle)
        patient.append((density,star,triangle))
    patient=np.asarray(patient)
    healthy_patients.append(patient)
    
disease_patients_path="C:\\Users\\wu\\Desktop\\TimeSeriesCoefficient\\Disease"
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
#######################################################################
#Classification

from sklearn import svm,neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold


loo = LeaveOneOut(len(y))
#lplo = LeavePLabelOut(labels, 2)
#ss = ShuffleSplit(5, n_iter=3, test_size=0.25,random_state=0)
kf = KFold(len(y), n_folds=15, indices=False)

C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=loo)
grid.fit(X, y)
print("The best classifier is: ", grid.best_estimator_)

count=0
for train, test in loo:    
    
    clf = svm.SVC(C=10,gamma=0.1)
    #clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights="distance")
    clf.fit(X[train], y[train])
    #print clf
    print clf.predict(X[test]),y[test]
    if clf.predict(X[test])==y[test]:
        count=count+1
    

print count


























  
