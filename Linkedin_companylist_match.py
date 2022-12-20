#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 00:53:20 2022

This code is to match the companies in linkedin spreadsheet to the main company spreadsheet
Remove those companies that do not exist in the main company spreadsheet

@author: zym_victoria

"""

import pandas as pd
import os

# Load files
# Compare long description company to company list
os.chdir('Please enter the path here')
folder_path = os.getcwd()
filename_1= 'Fail_company_Organized' #'Success_company_Organized' 
filename_2 = 'Fail Founder Extra Information 2' #'Success Founder Extra Information 2' 

f_company_list = pd.read_csv(folder_path+'/'+filename_1+'.csv')
f_founder_inf = pd.read_csv(folder_path+'/'+filename_2+'.csv')
f_founder_inf = f_founder_inf.dropna()
f_founder_inf = f_founder_inf.reset_index(drop=True)

f_CL_orgID= f_company_list['org_uuid'] # file company list 的 organization uuid
f_FI_orgID = f_founder_inf['org_uuid'] # file file information 的 organization uuid
f_FI_edu = f_founder_inf['Founder_education']

# =============================================================================
# # Check which companies are duplicated in the Compnay list csv
dup_CL_df = f_CL_orgID[f_CL_orgID.duplicated(keep=False)]
# =============================================================================

unique_FI_orgID = pd.Series(f_FI_orgID.unique())


org_search = unique_FI_orgID.isin(f_CL_orgID)
absent_company_idx = org_search[org_search==False].index # get the index of absent companies
present_company_idx = org_search[org_search==True].index # get the index of present companies
absent_companyID = unique_FI_orgID[list(absent_company_idx)] # get the name of absent companies
present_companyID = unique_FI_orgID[list(present_company_idx)] # get the name of present companies


# =============================================================================
# # If the unique length - absent company  = company list length
# abs_companyID_FI_idx = []
# for companyID in list(absent_companyID):
#     idx = f_FI_orgID[f_FI_orgID==companyID].index
#     abs_companyID_FI_idx += list(idx)
    
# new_FI_df = f_founder_inf.drop(abs_companyID_FI_idx)
# new_FI_df = new_FI_df.reset_index(drop=True)
# =============================================================================

# =============================================================================
# # If the unique length - absent company  != company list length
# 说明unique或者company list里面有对方没有的公司，把两个sheet里面都有的公司给分别找出来
present_companyID_FI_idx = []
present_companyID_CL_idx = []
for companyID in list(present_companyID):
    idx = f_FI_orgID[f_FI_orgID==companyID].index
    idx_2 = f_CL_orgID[f_CL_orgID==companyID].index
    present_companyID_FI_idx += list(idx)
    present_companyID_CL_idx += list(idx_2)

new_FI_df = f_founder_inf.iloc[present_companyID_FI_idx]
new_CL_df = f_company_list.iloc[present_companyID_CL_idx]

# =============================================================================

    
# Clean the content in founder information, replace [set()] and [{None}] by "unknown"
final_FI_df = new_FI_df.replace('[set()]','unknown')
final_FI_df = final_FI_df.replace('[{None}]','unknown')


# =============================================================================
# Check if the unique founder information ID is the same as the company list ID

check_FI_id = pd.Series(final_FI_df['org_uuid'].unique())
check_CL_id = pd.Series(new_CL_df['org_uuid'].unique())
check = check_FI_id == check_CL_id
check_False = check[check==False].index
# 检查了原始数据，false的原因是因为表格本来按照的是org name排序，他们的name相同但是id不同
# false是由于相同名字的公司id排反了
# 用orgID sort完应该就没事了

sorted_FI_df = final_FI_df.sort_values(by=['org_uuid'])
sorted_CL_df = new_CL_df.sort_values(by=['org_uuid'])
check_sFI_id = pd.Series(sorted_FI_df['org_uuid'].unique())
check_sCL_id = pd.Series(sorted_CL_df['org_uuid'].unique())
check2 = check_sFI_id == check_sCL_id
check2_False = check2[check2==False].index
# check2_False = 0 yeah~~

# =============================================================================

final_FI_df.to_csv('Fail_Founder_Inf_Organized.csv')






