#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:10:04 2022

This code is to combine the long and short description of the company list
For those have the long description, replace the 'short description' by 'long description'
For those do not have long descp, use the short ones.

@author: Yuanmin Zhang



"""

import pandas as pd
import os

# Load files
# Compare long description company to company list
os.chdir('Please enter the path here)
folder_path = os.getcwd()
filename_1= 'Please enter the company spread sheet name here (contain short description)'
filename_2 = 'Please enter the company spread sheet name here (contain long description)'

f_company_list = pd.read_csv(folder_path+'/'+filename_1+'.csv')
f_long_dsp = pd.read_csv(folder_path+'/'+filename_2+'.csv')


f_CL_orgname= f_company_list['org_name'] # file company list 的 organization name
f_LD_orgname = f_long_dsp['org_name'] # file long description 的 organization name

f_CL_shortDsp = f_company_list['short_description']
f_LD_longDsp = f_long_dsp['long_description']


# Search which company in the company list does not have long description
org_search = f_CL_orgname.isin(f_LD_orgname)
absent_company_idx = org_search[org_search==False].index # get the index of absent companies
present_company_idx = org_search[org_search==True].index
absent_company = f_CL_orgname[list(absent_company_idx)] # get the name of absent companies

# Create new company spreadsheet, with long description. For those don't have long description, use the short description
absent_company_inf = f_company_list.iloc[list(absent_company_idx)]
present_company_inf = f_company_list.iloc[list(present_company_idx)]

new_company_df = f_company_list.drop(list(absent_company_idx))
new_company_df = new_company_df.reset_index(drop=True)
new_company_df['short_description'] = f_long_dsp['long_description']
new_company_df = new_company_df.append(absent_company_inf)

# exclude nan
final_company_df = new_company_df.dropna()
final_company_df = final_company_df.reset_index(drop=True)

final_company_df.to_csv('Fail_company_Organized.csv')

## test
#t1 = new_company_df['org_name']
#t1 = t1.reset_index(drop=True)
#t2 = (t1==f_LD_orgname)

##check
#org_search_c = f_LD_orgname.isin(new_company_df['org_name'])
#absent_company_idx_c = org_search_c[org_search_c==False].index #确认过了都在


# Get rid of company with empty catagory list, and country 
# f_company_list = f_company_list.dropna()



