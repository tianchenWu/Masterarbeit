# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:40:31 2014

@author: wu
"""


import numpy as np
import igraph as ig
from sklearn import preprocessing
from sklearn.decomposition import PCA
import test.Global.globalvars as vars

def choose_best_lag():
    """
    Method based on autocorrelation function and partial autocorrelation
    Input: pandas dataframe time series data
    Output: an integer which is the best lag
    """

def dr(X,dim="mle"):
    """
    Dimensionality reduction
    Input: ndarray in shape (#sample,#feature)
    Output: ndarray in shape (#sample, #components)
    """
    pca = PCA(n_components=dim,whiten=False)
    X = pca.fit(X).transform(X)
    return X
    
    
    


def nan_filter(var_parameters,clinical_variables):
    """
    Filter all data records containing nan attribute
    input:var_parameters(2d array),clinical_variables(2d array)
    output:filtered var_parameters(2d array),clinical_variables(2d array)
    """
    nanID=[i for (i,x) in enumerate(clinical_variables) if np.isnan(x).any()]
    return [x for (i,x) in enumerate(var_parameters) if not i in nanID],[x for (i,x) in enumerate(clinical_variables) if not i in nanID]
    

def normalizer(var_parameters):
    """
    normalize var_parameters and clinical_variables using scale method
    input:var_parameters(2d array)
    output:filtered var_parameters(2d array)
    """
    """
    var_parameters_scaled = preprocessing.scale(var_parameters)
    clinical_variables_scaled=preprocessing.scale(clinical_variables)
    """
    min_max_scaler1 = preprocessing.MinMaxScaler()
    var_parameters_scaled= min_max_scaler1.fit_transform(var_parameters)

    
    
    return var_parameters_scaled
    
    
#check degree distribution        
            #for graph in g.graphlist[:17]:
            #    a=graph.degree_distribution()
            #    #plot hist
            #    xs, ys = zip(*[(left, count) for left, _, count in graph.degree_distribution().bins()])
            #    pylab.bar(xs, ys)
            #    pylab.show()
            #    print "oj"
"""
def analyse_multicolinearity(X):
    
    Print out eigenvalue of matrix X(transpose)X
    Input: 2d-array
    Output:array of eigenvalues
    
    #convert to matrix 
    X=np.matrix(X)
    #construct square matrix A=X(transpose)X
    A=np.dot(X.T,X)
    eigenvalues=np.linalg.eig(A)
    print "eigenvalues of XtransposeX is:"
    print eigenvalues[0]
"""
def plot_predict_error(errors):
    """
    Plot the prediction error by boxplot or histogram and fit a continous line on it
    """
    errors=[np.abs(x[0]) for x in errors]
    #print errors
    pl.figure()
    pl.boxplot(errors)
    fig=pl.figure()
    (mu, sigma) = norm.fit(errors)
    #fit histogram
    n, bins, patches = pl.hist(errors, bins=20, normed=True)
    fig.canvas.draw()
    y = mlab.normpdf(bins, mu, sigma)
    best_fit_line = pl.plot(bins, y, 'r--', linewidth=2)
    pl.xlabel("Error")
    pl.ylabel("Normalized Frequency")
    pl.title("Probability Density Function of Prediction Error")
    pl.grid(True)    


def floyd_warshall(graph):
    
    """
    Input:normalized graph
    Output:distance matrix
    --------------------------
    find the shortest weighted path between any of two vertices in the graph
    path length=1/normalized weight
    path length is in the range [0,number of vertices]
    --------------------------
    Algorithm:
    1 let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
    2 for each vertex v
    3    dist[v][v] ← 0
    4 for each edge (u,v)
    5    dist[u][v] ← w(u,v)  // the weight of the edge (u,v)
    6 for k from 1 to |V|
    7    for i from 1 to |V|
    8       for j from 1 to |V|
    9          if dist[i][j] > dist[i][k] + dist[k][j] 
    10             dist[i][j] ← dist[i][k] + dist[k][j]
    11         end if
    """
    
    dist_matrix=np.zeros((vars.N,vars.N))
    #initialize all entries off the diagonal in dist_matrix to be maximal path length
    #initialize all entries on the diagonal to be zero. distance to self is zero
    for index,x in np.ndenumerate(dist_matrix):
        i,j=index
        if i==j:
            dist_matrix[index]=1# this will be corrected back to zero at last step
        else:
            dist_matrix[index]=vars.N
    
    #for each edge (u,v)
    #    dist[u][v] ← w(u,v)  // the weight of the edge (u,v)
    for e in graph.es:
        i,j=e.tuple
        dist_matrix[i,j]=float(1)/float(e["weight"])
        if dist_matrix[i,j]>vars.N:
            dist_matrix[i,j]=vars.N
        dist_matrix[j,i]=dist_matrix[i,j]

        
    #update entry in dist_matrix
    for k in range(0,vars.N):
        for i in range(0,vars.N):
            for j in range(0,vars.N):
                if dist_matrix[i][k]*dist_matrix[k][j]<dist_matrix[i][j]:
                    dist_matrix[i][j]=dist_matrix[i][k]*dist_matrix[k][j]
                    dist_matrix[j][i]=dist_matrix[i][j]#keep the matrix symmetric
                #end if
    #end for
    for i in range(vars.N):
        dist_matrix[i,i]=0
    return dist_matrix
    

def avg_wpl(graph):
    """
    Average weighted path length for every vertex in the graph
    Input:shortest distance matrix
    Output:weighted average path length
    """
    dist_matrix=floyd_warshall(graph)
    avg=0
    for i in range(vars.N):
        i_sum=0
        for j in range(vars.N):
            i_sum=i_sum+dist_matrix[i,j]
        i_sum=float(i_sum)/float(vars.N-1)
        avg=avg+i_sum
    avg=float(avg)/float(vars.N-1)
    return avg


def parse_file_name(file_name):
    name_with_suffix=file_name.split("\\")[-1]
    name=name_with_suffix.split(".")[0]
    return name

def average_graph(graphlist,threshold,isWeighted):
    """
    Averaging graphs along the time series
    Average Strategy:
    If the graph is weighted: direct take the average over all time series
    If the graph is not weighted:
        	threshold 0.5:
         accept the edge if the probability of its appearance > threshold
         reject the edge if the probability of its appearance < threshold
    
    Input: a list of Graphs, threshold the probability of its appearance in time series
    Output: a igraph Object
    """
    prototype_graph=graphlist[0]
    is_weighted=prototype_graph.is_weighted()
    #labels=prototype_graph.vs["label"]
    #names=prototype_graph.vs["name"]
    adjacent_matrix=np.zeros((vars.N,vars.N))
    length=len(graphlist)
    THRESHOLD=threshold*length
    for g in graphlist:
        for e in g.es:
            _from,_to=e.tuple
            if is_weighted:
                adjacent_matrix[_from][_to]+=e["weight"]
            else:
                adjacent_matrix[_from][_to]+=1

            #keep the adjacent matrix symmetric
            adjacent_matrix[_to][_from]=adjacent_matrix[_from][_to]
    #average over time series
    if isWeighted:
        print "weighted average graph"
        f=np.vectorize(lambda x: x/float(length))
        adjacent_matrix=f(adjacent_matrix)
        averaged_graph=ig.Graph.Weighted_Adjacency(adjacent_matrix.tolist(), mode="undirected",attr="weight")
        assert averaged_graph.is_weighted(),"averaged graph should be weighted"
    if not isWeighted:
        #print "unweighted average graph"
        f=np.vectorize(lambda x: 1 if x> THRESHOLD else 0)
        adjacent_matrix=f(adjacent_matrix)
        averaged_graph=ig.Graph.Adjacency(adjacent_matrix.tolist(),mode="undirected")
        assert not averaged_graph.is_weighted(),"averaged graph should not be weighted"
    #averaged_graph.vs["label"]=labels
    #averaged_graph.vs["name"]=names
    #averaged_graph["st"]="averaged"
    return averaged_graph
    
def weighted_clustering_coefficient(g,vertice,weights="weight"):
    """
    vertex transitivities of the graph. Vertices with less than two neighbors require special treatment, 
    they will considered as having zero transitivity
    Calculation is based on the formula in paper
    A Weighted small world network measure for assessing functional connectivity
    formula (8)
    
    weight should be within interval [0,1]
    """
    k_i=g.degree(vertice)
    #print "k_i= %s" %k_i
    cc=0
    if k_i<=2:
        return cc
            
    neighbors=g.neighbors(vertice)
    print neighbors 
    #calculate for numerator 3*(sum w_ij,w_jh,w_hi),called part I,noted P1
    p1=sum([g[vertice,j]*g[vertice,h]*g[j,h] for j in neighbors for h in neighbors])  
    #calculate for the left part of denomerator (sum w_ih w_hj)
    p2=sum([g[vertice,h]*g[h,j] for j in neighbors for h in neighbors])
    if p1==0 and p2==0: return cc
    cc=float(3*p1)/float(2*p2+p1)
    return cc

def avg_wcc(g,weights="weight"):
    #max=np.max(np.array(g.es["weight"]))
    g.es["weight"]=g.es["weight"]/vars.MAX_WEIGHT
    sum_wcc=sum([weighted_clustering_coefficient(g,v,weights=weights) for v in g.vs])
    return float(sum_wcc)/float(len(g.vs))
    
if __name__=="__main__":

    g =ig.Graph()
    g.add_vertices(4)
    g.vs["label"]=["0","1","2","3",]
    g.add_edges([(0,1),(1,2),(0,3),(2,3)])
    g.es["weight"]=[0.1,0.5,1,0.5]
    #g.delete_edges((1,2),(3,4))
    layout=g.layout("kk")
    ig.plot(g,layout=layout)
    
    
    a=avg_wpl(g)
    print a
    
