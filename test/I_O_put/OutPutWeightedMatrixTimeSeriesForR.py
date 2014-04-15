# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:46:39 2014

@author: wu
"""



# -*- coding: utf-8 -*-
"""
Created on Sat Feb 08 22:08:30 2014

@author: wu
"""

import struct
import test.Global.globalvars as vars
import numpy as np
from test.I_O_put.Selector import *
from test.Global.HelpFunction import average_graph
import igraph as ig
from test.Global.HelpFunction import parse_file_name

#marke the relevant nodes as 1, others as 0
#[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1]

patientsNetworkDir="C:\\Users\\wu\\Desktop\\Patient"

selector=Selector("specified",weight_filter_method=vars.weight_filter_method)
data=selector.data
healthy_data,disease_data=data

for file_name,eeg in healthy_data:
    patientPath=patientsNetworkDir+"\\"+"Healthy\\"+parse_file_name(file_name).split("_")[1]+"_MAT"
    print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".bin"
        mat=np.asarray(g.get_adjacency(attribute="weight").data)#type list of list-> numpy array
        # create a binary file
        binfile = file(patientStampPath, 'wb')
        # and write out two integers with the row and column dimension
        header = struct.pack('2I', mat.shape[0], mat.shape[1])
        binfile.write(header)
        # then loop over columns and write each
        for i in range(mat.shape[1]):
            data = struct.pack('%id' % mat.shape[0], *mat[:,i])
            binfile.write(data)
        binfile.close()
    

#eye disease group
for file_name,eeg in disease_data:
    #graph=average_graph(eeg.graphlist,0.5)
    patientPath=patientsNetworkDir+"\\"+"Disease\\"+parse_file_name(file_name).split("_")[1]+"_MAT"
    print patientPath
    for i,g in enumerate(eeg.graphlist):
        if not os.path.isdir(patientPath):
            os.makedirs(patientPath)
        patientStampPath=patientPath+"\\"+str(i)+".bin"
        mat=np.asarray(g.get_adjacency(attribute="weight").data)#type list of list-> numpy array
        # create a binary file
        binfile = file(patientStampPath, 'wb')
        # and write out two integers with the row and column dimension
        header = struct.pack('2I', mat.shape[0], mat.shape[1])
        binfile.write(header)
        # then loop over columns and write each
        for i in range(mat.shape[1]):
            data = struct.pack('%id' % mat.shape[0], *mat[:,i])
            binfile.write(data)
        binfile.close()
    

