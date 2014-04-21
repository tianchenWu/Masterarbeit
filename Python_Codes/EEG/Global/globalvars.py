# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:57:55 2014

@author: wu
"""

import pandas as pd
import numpy as np
import os

global N
global rootDir
global graph_measurenames
global geye
global clinical_vairables
global SUBJECTS
global OPTOMETRY
global scale_method
global tsa_model_name
global weight_filter_method
global PERCENTILE
global DISCRETIZATION_STEP
global FREQUENCY
global debug

###################################################################
#Given facts 
###################################################################
debug=True
N=19
rootDir=os.path.split(os.path.dirname(__file__))[0]#/home/deathlypest/Masterarbeit/Python_Codes/EEG
tempDataDir=os.path.join(rootDir,"Data","tempData","clinical_variables.csv")
SUBJECTS=pd.read_csv(os.path.join(rootDir,"Data","klinische_Variablen","OVGU_FME_IMP_subjects_anonymized.csv"),sep=";")
OPTOMETRY=pd.read_csv(os.path.join(rootDir,"Data","klinische_Variablen","OVGU_FME_IMP_optometry_nan2k(3)-nn_eyes_wght.csv"),sep=",")

#OPTOMETRY=pd.read_csv(rootDir+"\\klinische_Variablen\\OVGU_FME_IMP_optometry.csv")
#OPTOMETRY.columns=["ID","eye","condition","1","2","3","4","5","6","7","SP_FT","9","10","11","12","13","VA_N_log","VA_F_log"]
#OPTOMETRY=OPTOMETRY[OPTOMETRY.eye=="OD"][~np.isnan(OPTOMETRY["VA_N_log"])][~np.isnan(OPTOMETRY["VA_F_log"])][~np.isnan(OPTOMETRY["SP_FT"])]


###################################################################
#Experiement Setting                                              
###################################################################

#geye="OD"  #right eye
#scale method
graph_measurenames=["density","clustering coefficient","average path length"]
clinical_variable_names=["SP_FT"]
#validation_method="leave one out" this is fixed not variable
tsa_model_name="VAR"
weight_filter_method="binarilize"
PERCENTILE=30   #the percentile of data being pruned, PERCENTILE=0 means no edge is pruned      
DISCRETIZATION_STEP=0.071
FREQUENCY="alpha"
MAX_WEIGHT=np.floor(1/DISCRETIZATION_STEP)

###################################################################
#  Can be choosen from                                            
###################################################################



#weight_filter_method="binarilize"
#weight_filter_method="discretize"
#weight_filter_method="original"

#graph_measurenames=["TWS"] #traditional small world model
#graph_measurenames=["density","clustering coefficient","average path length"]
#graph_measurenames=["RG_p"]
#graph_measurenames=["degree sequence"]
#graph_measurenames=["WRG_p"]


#tsa_model_name="VAR"
#tsa_model_name="AR"
#tsa_model_name="SM"

#rootDir="C:\\Users\\deathlypest\\Dropbox\ma_tianchen_wu\\daten"

########################################################
#Data 1
########################################################
#condition                                                                  
#detection accuracy in HRP defective visual field sectors [%]               
#detection accuracy in whole HRP visual field [%]                           
#fixation accuracy in HRP [%]                                               
#false positive reactions in HRP [%]                                       
#reaction time in HRP [ms]                                                  
#reaction time in HRP defective visual field sectors [ms]                   
#fixation accuracy in static perimetry [%]                                  
#foveal threshold in static perimetry [dB]                                  
#mean threshold in static perimetry (whole 30 degrees visual field) [dB]    
#mean eccentricity in kinetic perimetry [degrees]                           
#added eccentricity in kinetic perimetry                                   
#near vision (decimal scale)                                               
#far vision (decimal scale)                                                 
#near vision (LogMAR scale)                                                
#far vision (LogMAR scale)                                                  



#######################################################
#Data 3-nearst neighbour
#######################################################
#HRP_RT:reation time, better not to use them
#HRP_RT_g:reation time
#HRP_FA:Fixation rate in HRP
#HRP_FP:False Positive in HRP

#ID
#HRP_DA_g
#HRP_DA
#HRP_DA_CMF
#HRP_DA_g_CMF
#SP_FA
#SP_FT
#SP_MT
#KP_ME
#KP_AE
#VA_N_dec
#VA_F_dec
#VA_N_log
#VA_F_log