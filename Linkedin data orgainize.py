# -*- coding: utf-8 -*-
"""
Spyder Editor

The code is to extract useful information from the Linkedin profile (stored in json)
The extracted information include:
    * Founder description
    * Founder education
    * Founder previous employer company description
    * Founder previous jobtitles 
    
@author: Yuanmin Zhang
"""
import pandas as pd
import json
import os


os.chdir('Please enter the path here)
folder_path = os.getcwd()
filename = 'Please enter the spread sheet name here (contain founder linkedin details in json format)'
#lpf : Linkedin Profile
l_pf = pd.read_csv(folder_path+'/'+filename+'.csv')



lpf_linkedin= l_pf['linkedin_url']
lpf_json = l_pf['json_string']

len(l_pf)
Founder_description = []
Founder_education = []
Founder_employerCompany = []
Founder_jobTitle = []

for i in range(len(l_pf)):
#for i in range(14,16):
    # Check the json string
    lpf_json_i = lpf_json[i] #type: string
    if type(lpf_json[i])!=str:
        continue
    lpf_json_i = json.loads(lpf_json_i) #type: dict
    lpf_data = lpf_json_i['data'] # type: list containing a dict
    if len(lpf_data) !=0:
        lpf_data_dic = lpf_data[0] # type: dict
    else: 
        continue
    
            
    #Obtain founder description: (dscp: description)
    alldscp_inf = lpf_data_dic.get('allDescriptions')
    dscp_inf = lpf_data_dic.get('description')
    if alldscp_inf != None:
        dscp_inf = alldscp_inf
    elif dscp_inf == None:
        dscp_inf = []
    #Convert the list into set, 让所有的data都可以挤在一个cell里
    if type(dscp_inf)==list:
        dscp_inf_set = set(dscp_inf)
    else:
        dscp_inf_set = dscp_inf
    
    
    #Obtain founder education: (edu: education)
    edu_inf = []
    edu_data = lpf_data_dic.get('educations')
    if edu_data !=None:
        for each_edu in edu_data:
            institution_inf = each_edu.get('institution')
            if institution_inf != None:
                inf_1 = institution_inf.get('name')
                inf_2 = institution_inf.get('summary')
                if inf_2!=None:
                    # inf_2 = institution_inf['summary']
                    # Group the education information together 
                    edu_inf.append(inf_1+' is a '+inf_2)
                else:
                    edu_inf.append(inf_1)
    #Convert the list into set
    edu_inf_set = set(edu_inf)
    #print(edu_inf)

    
    #Obtain founder employment details and job titles: (emp: employment)
    emp_company_inf = []  # founder previous & current companies
    emp_jobtitle_inf = [] # founder previous & current job titles 
    emp_data = lpf_data_dic.get('employments')
    if emp_data != None:
        for emp_history in emp_data:
            inf_employer = emp_history.get('employer')
            if inf_employer != None:
                inf_employer_name = inf_employer['name']        
                inf_employer_summary = inf_employer.get('summary')
                if inf_employer_summary!=None:
                    # Group the employer company information together 
                    emp_company_inf.append(inf_employer_name + " is a " + inf_employer_summary)
                else:
                    emp_company_inf.append(inf_employer_name)
        
            inf_jobtitle = emp_history.get('title')
            if inf_jobtitle != None:
                emp_jobtitle_inf.append(inf_jobtitle)  
            
    #Convert the list into set
    emp_company_inf_set = set(emp_company_inf)
    emp_jobtitle_inf_set = set(emp_jobtitle_inf)
    #print(emp_company_inf)
    #print(emp_jobtitle_inf)
    
    #Add to the final dataset
    Founder_description.append([dscp_inf_set])
    Founder_education.append([edu_inf_set])
    Founder_employerCompany.append([emp_company_inf_set])
    Founder_jobTitle.append([emp_jobtitle_inf_set])
    
    
#Organize the data into a dictionary    
output_dict = {'Founder_description':Founder_description, 'Founder_education':Founder_education,'Founder_employerCompany':Founder_employerCompany,'Founder_jobTitle':Founder_jobTitle}
#output data frame
output_df = pd.DataFrame(output_dict)
final_df = pd.concat([lpf_linkedin,output_df],axis=1) # add linkedlin to the first column (留下linkedlin只是为了找到主数据)
#Save the dataframe to csv
final_df.to_csv('Founder Extra Information2.csv')
#final_df.to_csv('test_file_2.csv')