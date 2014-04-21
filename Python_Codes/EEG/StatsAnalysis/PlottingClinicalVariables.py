# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:58:29 2014

@author: wu
"""
import matplotlib.pylab as pl
import pandas as pd
import numpy as np
import test.Global.globalvars as vars


def scatter_hist_plot(X,Y,X_name,Y_name,title):
    # definitions for the axes
    nullfmt   = pl.NullFormatter()         # no labels
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    
    axScatter = pl.axes(rect_scatter)
    axHistx = pl.axes(rect_histx)
    axHisty = pl.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(X, Y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(X)), np.max(np.fabs(Y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim( (-lim, lim) )
    axScatter.set_ylim( (-lim, lim) )
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(X, bins=bins)
    axHisty.hist(Y, bins=bins, orientation='horizontal')
    
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
                  
    axScatter.set_xlabel(X_name)
    axScatter.set_ylabel(Y_name)
    pl.title(title)


group1=["VA_N_log","VA_F_log","SP_FT"]
G=vars.OPTOMETRY[group1]
"""
group1=["VA_F_log"]
G=vars.OPTOMETRY[group1]
fig=pl.figure()
G.boxplot()
print G
"""

def plot_cv_against_eachother():
    
    #plot clinical variables against each other
    
    cv_pairs=[(x,y) for x in G.columns for y in G.columns if x is not y]
    print cv_pairs
    cv_pairs=[cv_pairs[i] for i in [0,1,3]]
    print cv_pairs
    rows=len(cv_pairs)
    for i,(x,y) in enumerate(cv_pairs):
        fig=pl.subplot(rows,1,i)
        pl.xlabel(x)
        pl.ylabel(y)
        X,Y=G[x],G[y]
        pl.scatter(X,Y)
        pl.tight_layout()
        pl.grid()
        #xlim=np.max(X)
        #ylim=np.max(Y)
        #pl.xlim(xlim)
        #pl.ylim(ylim)
        
        
        
    #clinical_variable=vars.OPTOMETRY["VA_N_log"]
    #pl.scatter(np.arange(len(clinical_variable)),clinical_variable)

def cv_boxplot():
    for x in group1:
        fig=pl.figure()
        X=G[x]
        pl.boxplot(X,0,'gD')
        pl.title(x)
        pl.grid(True)

#print vars.OPTOMETRY
plot_cv_against_eachother()
#cv_boxplot()