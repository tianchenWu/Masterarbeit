# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:11:09 2014

@author: wu
"""
import matplotlib.pylab as pl
import pandas as pd
import numpy as np
import test.Global.globalvars as vars


X=vars.OPTOMETRY["SP_FT"]
fig=pl.figure()
pl.boxplot(X,0,'gD')
pl.title("SP_FT")
pl.grid(True)


