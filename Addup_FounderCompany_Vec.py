#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:00:32 2022

Add up the Company vector and the Founder information vectors 

@author: Yuanmin Zhang
"""

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
import math
import nltk
from numpy import dot
from numpy.linalg import norm

# Load files
# Compare long description company to company list
os.chdir('Please enter the path here')
folder_path = os.getcwd()
filename_1= 'Fail_Founder_sumVector'  
filename_2= 'Fail_company_sumVector' 

f_1 = pd.read_csv(folder_path+'/'+filename_1+'.csv')
f_2 = pd.read_csv(folder_path+'/'+filename_2+'.csv')

f_orgID = f_1['org_uuid']
f_orgName = f_1['org_name']
f_vec = f_1['Total features'] 

f_company_vec = f_2['Total Company features']
f_company_ID = f_2['org_uuid']
f_company_name = f_2['org_name']

unique_orgID = pd.Series(f_orgID.unique())


org_search = f_company_ID.isin(unique_orgID)
absent_company_idx = org_search[org_search==False].index # get the index of absent companies
absent_companyID = f_company_ID[list(absent_company_idx)] # get the name of absent companies





companyfounders_vec_list = []
companyinf_vec_list = []
total_features_vec_list = []
i=0

for i in range(len(unique_orgID)):
    
    ID = pd.Series(unique_orgID[i])
    duplicate_orgID_search = f_orgID.isin(ID)
    duplicate_orgID_idx = duplicate_orgID_search[duplicate_orgID_search==True].index
    duplicate_vec = f_vec[list(duplicate_orgID_idx)]
    
    #这里得找办法吧list里面的string变成vector/ array
    companyfounders_vec = np.empty(768)
    for each_vec in duplicate_vec:
        vec_str = each_vec.replace('[','').replace(']','')
        vec_str = vec_str.split(', ')
        vec_float = [float(element) for element in vec_str]
        vec_array = np.array(vec_float)
        
        companyfounders_vec += vec_array
        
    
    
    company_vec_str = f_company_vec[i].replace('[','').replace(']','')
    company_vec_str = company_vec_str.split(', ')
    company_vec_float = [float(element) for element in company_vec_str]
    company_vec_array = np.array(company_vec_float)
    
    total_features_vec = companyfounders_vec + company_vec_array
    
    
    companyfounders_vec_list.append(companyfounders_vec)
    companyinf_vec_list.append(company_vec_array)
    total_features_vec_list.append(total_features_vec)
    

f_ID_list = f_company_ID.tolist()
f_org_list = f_company_name.tolist() 
output_dict_1 = {'org_uuid':f_ID_list, 'org_name':f_org_list , 'Sum company founder features': companyfounders_vec_list}
output_dict_2 = {'org_uuid':f_ID_list, 'org_name':f_org_list , 'Total comany and founder features' : total_features_vec_list}
#output data frame
final_df_1 = pd.DataFrame(output_dict_1)
final_df_2 = pd.DataFrame(output_dict_2)
#Save the dataframe to csv
#final_df_1.to_csv('Fail_CombinedFounder_sumVector.csv')
#final_df_2.to_csv('Fail_CompanyFounder_totalsumVector.csv')
        

dict_output = {'Cp1':companyfounders_vec_list[0]}
df = pd.DataFrame(dict_output)

for i in range(1,len(f_ID_list)):
    name = "Cp" + str(i+1)
    df[name] = companyfounders_vec_list[i]

#df["State"] = ['Success']*768


df.to_csv('Fail_Founder_totalsumVector_Expand.csv')
    
    
    
    
    
    
    
    
    
