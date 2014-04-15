

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:04:33 2014

@author: deathlypest
"""
import numpy as np


#read in file
f = open("C:\\Users\\wu\\spyder_workspace\\HurryUp2\\R Scripts\\AveragedERGM_0.3_0.3_kstar(2)triangle_coefficients.txt")
is_healthy=False
is_disease=False
healthy_patients=[]
disease_patients=[]
count=1
print f
for line in f.readlines():
    print "hi"
    print line
    if is_healthy and line[:-1]!="Healthy" and line[:-1]!="Disease":
        record=[float(i) for i in line.split() if i!=" "]
        #print record
        healthy_patients.append(record)
        
    if is_disease and line[:-1]!="Healthy" and line[:-1]!="Disease":
        record=[float(i) for i in line.split() if i!=" "]
        #print record
        disease_patients.append(record)
        
    if line[:-1]=="Healthy":
        #print line
        is_healthy=True
    if line[:-1]=="Disease":
        #print line
        is_healthy=False
        is_disease=True
        
healthy_patients=np.asarray(healthy_patients)
disease_patients=np.asarray(disease_patients)

#print healthy_patients
#print disease_patients    
f.close()

#analyze the distinguishability of the parameters
print np.mean(healthy_patients,axis=0)
#disease_patients
print np.mean(disease_patients,axis=0)
print np.std(healthy_patients,axis=0)
print np.std(disease_patients,axis=0)




"""

#visualize data set

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xs_h=[i[0] for i in healthy_patients ]
ys_h=[i[1] for i in healthy_patients ]
zs_h=[i[2] for i in healthy_patients ]

xs_d=[i[0] for i in disease_patients ]
ys_d=[i[1] for i in disease_patients ]
zs_d=[i[2] for i in disease_patients ]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs_h, ys_h, zs_h, c="r", marker="o")
ax.scatter(xs_d, ys_d, zs_d, c="b", marker="^")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.scatter(ys_h, zs_h, c="r", marker="o")
plt.scatter(ys_d, zs_d, c="b", marker="^")
plt.show()
"""


#preprocessing
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
healthy_patients_minmax = min_max_scaler.fit_transform(healthy_patients)
disease_patients_minmax= min_max_scaler.fit_transform(disease_patients)

#print healthy_patients_minmax
#print disease_patients_minmax

#combine healthy_patients and disease_patients
#target healthy patient as 1, disease patient as 0
X=np.concatenate((healthy_patients,disease_patients),axis=0)
#print len(X)
y=np.zeros(30)
for i in range(15):
    y[i]=1
#print y

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














