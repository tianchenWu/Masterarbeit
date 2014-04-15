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
import matplotlib.pylab as pl
from test.Evaluation.Evaluation import Evaluation
import test.Global.globalvars as vars
from test.I_O_put.Selector import Selector
from test.Global.HelpFunction import normalizer,dr


def analyze_eval_result(differences):
    errors=np.asarray([np.abs(x) for x in differences])
    mean=errors.mean(axis=0)
    std=errors.std(axis=0)
    print "absolute mean:  %s, standard deviation:  %s" %(mean,std)
    return mean,std

#read in data
root="C:\\Users\\wu\\spyder_workspace\\HurryUp2\\TempData\\TimeSeriesCoefficientRegression"
cv_table_path="C:\\Users\\wu\\spyder_workspace\\HurryUp2\\TempData\\cv_table.txt"
#read in cv_table
cv_table=pd.DataFrame.from_csv(path=cv_table_path)
column_names=list(cv_table.columns.values)[:-1]#exclude KEY column
print column_names
"""
For each clinical variable
"""
for column in column_names:
    patients=[]
    for file in listdir(root):
        #print file
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
        print file_name
        if [x for x in cv_table["KEY"]==file_name if x==True]==[]:
            continue
        print cv_table[cv_table["KEY"]==file_name][column]
        cv=float(cv_table[cv_table["KEY"]==file_name][column])
        y.append(cv)
    print len(y)
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
        n=original_raw_data.shape[0]
        #the form of data_each_measure is 1d numpy array
        #fig,axis_list=plt.subplots(nrows=n, sharex=True)
        for i,data_each_measure in enumerate(original_raw_data):
            #axis_list[i].plot(arange(len(data_each_measure)),data_each_measure)
            filtered_data_each_measure=savitzky_golay(data_each_measure, window_size=31, order=7)
            raw_data.append(filtered_data_each_measure)
            #axis_list[i].plot(arange(len(data_each_measure)),filtered_data_each_measure,"r")
        patient=np.asarray(raw_data).swapaxes(1,0)
        ################################################################################
        
        model=VAR(patient)
        result=model.fit(1,trend="nc")
        
        coefs=(result.coefs).flatten()
        X.append(coefs)
        """
        coefs_matrix=[]
        for coef in coefs:
            matrix=np.matrix(coef)
            coefs_matrix.append(matrix)
        coefs_matrix=np.asarray(coefs_matrix)
        #take eigenvalues of coef
        eigenvalue_list=[]
        for matrix in coefs_matrix:
            eigenvalues,eigenvectors=np.linalg.eig(matrix)
            eigenvalue_list.append(eigenvalues)
        coefs=np.asarray(eigenvalue_list).flatten()
        X.append(coefs)
        """
    X=np.asarray(X)
    X=normalizer(X)
    X=dr(X,dim="mle")
    y=np.asarray(y)
    #y=y*100
    
    ########################################################################
    from sklearn import preprocessing
    #preprocessing
    min_max_scaler1 = preprocessing.MinMaxScaler()
    X= min_max_scaler1.fit_transform(X)
    min_max_scaler2 = preprocessing.MinMaxScaler()
    y=np.asarray([[i]for i in y.tolist()])
    y= min_max_scaler2.fit_transform(y)
    
    
    #######################################################################
    #Regression
    
    from sklearn import linear_model,svm,tree,neighbors
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import LeaveOneOut,ShuffleSplit,KFold,LeavePLabelOut
    
    
    #loo = LeaveOneOut(len(y))
    #lplo = LeavePLabelOut(labels, 2)
    #ss = ShuffleSplit(5, n_iter=3, test_size=0.25,random_state=0)
    #kf = KFold(len(y), n_folds=10, indices=False)
    
    errors=[]
    N=10
    K=15
    for i in range(N):
        kf = KFold(len(y), n_folds=K, indices=False,shuffle=True)

        differences=[]
        for train, test in kf:    
            
            #clf = linear_model.LinearRegression()
            #clf=linear_model.Ridge(2)
            #clf=svm.SVR(C=1,gamma=1)
            #clf = tree.DecisionTreeRegressor()
            clf=neighbors.KNeighborsRegressor(n_neighbors=1, weights="uniform")
            
            clf.fit(X[train], y[train])
            predicted=clf.predict(X[test])
            print predicted,y[test]
            z=[y[test][i]-predicted[i] for i in range(len(predicted))]                        
            differences.extend(z)    
            
        analyze_eval_result(differences)
        error=np.asarray([np.abs(x) for x in differences])
        error=np.mean(error)
        errors.append(error)
    errors=np.asarray(errors)
    #boxplot errors
    figure()

    
    #draw random guess line
    rad_guess=np.mean(y)
    rad_guess_errors=[np.abs(rad_guess-c) for c in y]
    averaged=np.mean(np.asarray(rad_guess_errors))
    
    #draw benchmark line RG_p
    vars.clinical_variable_names=[column]
    vars.tsa_model_name="AR"
    vars.graph_measurenames=["RG_p"]
    vars.weight_filter_method="binarilize"

    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    eval=Evaluation(selector)
    GR_p=eval.evaluate()
    

    """
    #draw benchmark line WRG_p
    vars.clinical_variable_names=[column]
    vars.tsa_model_name="AR"
    vars.graph_measurenames=["WRG_p"]
    vars.weight_filter_method="discretize"

    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    eval=Evaluation(selector)
    WGR_p=eval.evaluate()


    #draw benchmark line clustering_coefficient, average_path_length,density
    vars.clinical_variable_names=[column]
    vars.tsa_model_name="VAR"
    vars.graph_measurenames=["density","clustering coefficient","average path length"]
    vars.weight_filter_method="binarilize"

    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    eval=Evaluation(selector)
    BDM=eval.evaluate()



    #draw benchmark line weighted clustering_coefficient, average_path_length,density
    vars.clinical_variable_names=[column]
    vars.tsa_model_name="VAR"
    vars.graph_measurenames=["density","clustering coefficient","average path length"]
    vars.weight_filter_method="discretize"
    selector=Selector("all",weight_filter_method=vars.weight_filter_method)
    eval=Evaluation(selector)
    WBDM=eval.evaluate()

    data=[errors,averaged,GR_p,WGR_p,BDM,WBDM]
    """
    data=[errors,averaged,GR_p]
    
    pl.boxplot(data,0,'')    
    
    pl.ylabel("Error")
    pl.title(column)
    pl.grid(True)

    #pl.xticks((1,2,3,4,5,6),("ERGM","Average Prediction","RG_p","WRG_p","Binary Descriptive Model","Weighted Descriptive Model"))
    pl.xticks((1,2,3),("ERGM","Average Prediction","RG_p"))
    














  
