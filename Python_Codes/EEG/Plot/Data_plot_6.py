# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:17:40 2014

@author: wu
"""

group1=["HRP_FA","SP_FA"]
G=vars.OPTOMETRY[group1]


def plot_cv_against_eachother():
    
    cv_pairs=[(x,y) for x in G.columns for y in G.columns if x is not y]
    print cv_pairs
    cv_pairs=[cv_pairs[1]]
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
    

plot_cv_against_eachother()