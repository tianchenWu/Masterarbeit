# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 19:30:26 2014

@author: wu
"""
from os import listdir
import numpy as np
from statsmodels.tsa.vector_ar.var_model import *
from test.Filter.LowPassFilter import savitzky_golay
import pandas as pd

#read in data
root="C:\\Users\\wu\\Desktop\\TimeSeriesCoefficientRegression"
cv_table_path="C:\\Users\\wu\\Desktop\\R Scripts\\cv_table.txt"
#read in cv_table
cv_table=pd.DataFrame.from_csv(path=cv_table_path)

patients=[]
for file in listdir(root):
    
    file_fullpath=root+"\\"+file
    f=open(file_fullpath)
    patient=[]
    for line in f.readlines():
        density,star,triangle=line.split(" ")
        triangle,_=triangle.split("\n")
        density,star,triangle=float(density),float(star),float(triangle)
        patient.append((density,star,triangle))
    patient=np.asarray(patient)
    patients.append((file.split(".")[0],patient))

y=[]
for file_name,_ in patients:
    cv=float(cv_table[cv_table["0"]==file_name]["1"])
    y.append(cv)
patients=[i for _,i in patients]


#########################################################################
#processing time series
#VAR model
X=[]
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

########################################################################
from sklearn import preprocessing
#preprocessing
min_max_scaler1 = preprocessing.MinMaxScaler()
X= min_max_scaler1.fit_transform(X)




#######################################################################
#Regression

from sklearn import linear_model,svm,tree,neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold,LeavePLabelOut


loo = LeaveOneOut(len(y))
#lplo = LeavePLabelOut(labels, 2)
#ss = ShuffleSplit(5, n_iter=3, test_size=0.25,random_state=0)
#kf = KFold(len(y), n_folds=10, indices=False)


#select paramter for SVR
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVR(), param_grid=param_grid, cv=kf)
grid.fit(X, y)
print("The best classifier is: ", grid.best_estimator_)


differences=[]
for train, test in loo:    
    
    #clf = linear_model.Lasso (alpha=0.001)
    clf=svm.SVR(C=1,gamma=1)
    #clf = tree.DecisionTreeRegressor()
    #clf=neighbors.KNeighborsRegressor(n_neighbors=1, weights="uniform")
    
    clf.fit(X[train], y[train])
    predicted=clf.predict(X[test])
    print predicted,y[test]
    z=[y[test][i]-predicted[i] for i in range(len(predicted))]                        
    differences.extend(z)

def analyze_eval_result(differences):
        errors=np.asarray([np.abs(x) for x in differences])
        mean=errors.mean(axis=0)
        std=errors.std(axis=0)
        print "absolute mean:  %s, standard deviation:  %s" %(mean,std)
        return mean,std
        


analyze_eval_result(differences)


























  

