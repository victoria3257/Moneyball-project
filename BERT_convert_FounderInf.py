#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:11:39 2022

Use BERT to convert the paragraphs or phrases in each cell into a vector

This code applies to the founder information spreadsheet:

@author: Yuanmin Zhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy
import nltk
import math
from numpy import dot
from numpy.linalg import norm


# =============================================================================
# # load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
# from_pretrained will fetch the model from the internet
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,)
# output_hidden_states = True: Whether the model returns all hidden-states.
                                  
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
# =============================================================================


# =============================================================================
""" Define Some Functions"""
# # clean the sentence tokens
def clean_sentences(text):
    """Make text lowercase, remove punctuation and remove words containing numbers."""
    
    # replace webpage
    # todo: this part can be further fine tuned
    patterns = ['http:', 'www.']
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    text = re.sub("[\(].*?[\)]", "", text)  # clear all the contents within parenthese
    text = re.sub(r'[^\w,\,\.\ ]', '', text)  # clear all things except underscore, alphanumeric, comma and stop.
    text = text.lower()  # lower all text
    
    
    return text


def deepclean_sentences(text):
    """ Split the sentences in the list, add stop between them and rejoin them into a single paragraph"""
    split_text = text.split('\',')
    c_sentences = [clean_sentences(line) for line in split_text] # clean the sentences
    c_joined = '. '.join(c_sentences)  # is a string
    
    return c_joined


def separate_paragraphs(text):
    """"Separate too long paragraphs into two parts"""
    
    n = math.ceil(len(text)/2400)
    
    nltk_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentences = nltk_tokenizer.tokenize(text)
    length = len(sentences)
    l1 = math.floor(length/n)
    l2 = math.floor(length*2/n)
    sentences_1 = sentences[0:l1]
    sentences_2 = sentences[l1:l2]
    sentences_1 = ' '.join(sentences_1)
    sentences_2 = ' '.join(sentences_2)
    
    return sentences_1, sentences_2


def get_SentenceVector(hidden_states):
    """Average the second to last hidden later of each token."""
    # `hidden_states` has shape [13 x 1 x #tokens x 768] = [#layer, # batches, #token, # features]
    # `token_vecs` is a tensor with shape [#tokens x 768]
    token_vecs = hidden_states[-2][0]
    
    # Calculate the average of all token vectors.
    embedding = torch.mean(token_vecs, dim=0) # is a vector if length 768
    
    return embedding


def vocabulary_vector(vocabulary,token_list,token_vecs_sum):
    """ Search the vocabulary in the token list, use the index to extract the vector """
    """ Use 'vocabulary' instead of 'word' here is just to avoid repetition"""
    """ Return vocabulary vec """ 
    vocab_tkn = tokenizer.tokenize(vocabulary)
    vocab_vec = torch.zeros([len(token_vecs_sum[0])])
    for each_tkn in vocab_tkn:
        idx = token_list.index(each_tkn)
        vocab_vec += token_vecs_sum[idx]
    return vocab_vec


def get_WordVector(hidden_states, words, token_list):
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.[#layer, # batches, #token, # features]
    token_embeddings = torch.stack(hidden_states, dim=0)
    
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    
    # Swap dimensions 0 and 1. So the tensor is in the form of [#token,#layer,#features]
    token_embeddings = token_embeddings.permute(1,0,2)
     
    # Stores the token vectors, with shape [token number x 768]
    token_vecs_sum = []
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor ([#layer, # features])
        
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    
    words_vec = torch.zeros([len(token_vecs_sum[0])])
    for word in words.split(" "):        
        word_vec = vocabulary_vector(word,token_list,token_vecs_sum)
        words_vec += word_vec 
        
    return words_vec

# =============================================================================


# Load files
# Compare long description company to company list
os.chdir('Please enter the path here)
folder_path = os.getcwd()
filename_1= 'Fail_Founder_Inf_Organized_cleanup'  #'Fail_company_Organized_cleanup'

f_founder_inf = pd.read_csv(folder_path+'/'+filename_1+'.csv')

f_ID = f_founder_inf['org_uuid']
f_orgName = f_founder_inf['org_name']

f_dsp = f_founder_inf['Founder_description'] 
f_edu= f_founder_inf['Founder_education'] 
f_emp = f_founder_inf['Founder_employerCompany'] 
f_jobTitle= f_founder_inf['Founder_jobTitle']


# dsp_hiddenstates_list = []
# edu_hiddenstates_list = []
# emp_hiddenstates_list = []
# jobTitle_hiddenstates_list = []

dsp_embedding_list = []
edu_embedding_list = []
emp_embedding_list = []
jobTitle_embedding_list = []
total_embedding_list = []

error_idx = []
each_company = 0

for each_company in range(len(f_dsp)):
#for each_company in range(200,255):
    
    error = False 
    c_dsp_separate = False
    c_emp_separate = False
    
    dsp_hiddenstates = []
    edu_hiddenstates = []
    emp_hiddenstates = []
    jobTitle_hiddenstates = []
    
    dsp_embedding = []
    edu_embedding = []
    emp_embedding = []
    jobTitle_embedding = []
    total_embedding = []
    

    # clean the sentences
    c_dsp = clean_sentences(f_dsp[each_company])
    c_edu = deepclean_sentences(f_edu[each_company])
    c_emp = deepclean_sentences(f_emp[each_company])
    c_jobTitle = deepclean_sentences(f_jobTitle[each_company])
    
    if len(c_dsp) > 2500:
        [c_dsp,c_dsp_2] = separate_paragraphs(c_dsp)
        c_dsp_separate = True
    
        
    if len(c_emp) > 2500:
        [c_emp,c_emp_2] = separate_paragraphs(c_emp)
        c_emp_separate = True
    

    # Adjust the format of sentences for BERT model
    # Mark each sentence with starting [CLC] and end with [SEP]
    marked_dsp = "[CLS] " + c_dsp + " [SEP]"
    marked_edu = "[CLS] " + c_edu + " [SEP]"
    marked_emp = "[CLS] " + c_emp + " [SEP]"
    marked_jobTitle = "[CLS] " + c_jobTitle + " [SEP]"

    dsp_tokens = tokenizer.tokenize(marked_dsp) # dsp_tkn: tokenized description
    edu_tokens = tokenizer.tokenize(marked_edu)
    emp_tokens = tokenizer.tokenize(marked_emp)
    jobTitle_tokens = tokenizer.tokenize(marked_jobTitle)
    
    if len(dsp_tokens) > 512:
        dsp_tokens = dsp_tokens[0:511] + ['SEP']
    elif len(emp_tokens) > 512:
        emp_tokens = emp_tokens[0:511] + ['SEP']
    
    # Map the token strings to their vocabulary indeces.
    idx_dsp_tokens = tokenizer.convert_tokens_to_ids(dsp_tokens)
    idx_edu_tokens = tokenizer.convert_tokens_to_ids(edu_tokens)
    idx_emp_tokens = tokenizer.convert_tokens_to_ids(emp_tokens)
    idx_jobTitle_tokens = tokenizer.convert_tokens_to_ids(jobTitle_tokens)


    # Mark each of the tokens as belonging to sentence "1".
    # The input is only single sentence
    segments_idf_dsp = [1] * len(dsp_tokens)
    segments_idf_edu = [1] * len(edu_tokens)
    segments_idf_emp = [1] * len(emp_tokens)
    segments_idf_jobTitle = [1] * len(jobTitle_tokens)
    
    # Convert inputs to PyTorch tensors
    dsp_tokens_tensor = torch.tensor([idx_dsp_tokens]) # type: torch.Tensor
    dsp_segments_tensors = torch.tensor([segments_idf_dsp]) # type: torch.Tensor
    edu_tokens_tensor = torch.tensor([idx_edu_tokens]) # type: torch.Tensor
    edu_segments_tensors = torch.tensor([segments_idf_edu]) # type: torch.Tensor
    emp_tokens_tensor = torch.tensor([idx_emp_tokens]) # type: torch.Tensor
    emp_segments_tensors = torch.tensor([segments_idf_emp]) # type: torch.Tensor
    jobTitle_tokens_tensor = torch.tensor([idx_jobTitle_tokens]) # type: torch.Tensor
    jobTitle_segments_tensors = torch.tensor([segments_idf_jobTitle]) # type: torch.Tensor
    
    
    if c_dsp_separate:
        
        marked_dsp2 = "[CLS] " + c_dsp_2 + " [SEP]"
        dsp2_tokens = tokenizer.tokenize(marked_dsp2)
        
        if len(dsp2_tokens) > 512:
            dsp2_tokens = dsp2_tokens[0:511] + ['SEP']
        
        idx_dsp2_tokens = tokenizer.convert_tokens_to_ids(dsp2_tokens)
        segments_idf_dsp2 = [1] * len(dsp2_tokens)

        dsp2_tokens_tensor = torch.tensor([idx_dsp2_tokens]) # type: torch.Tensor
        dsp2_segments_tensors = torch.tensor([segments_idf_dsp2]) # type: torch.Tensor
    
    if c_emp_separate:
        
        marked_emp2 = "[CLS] " + c_emp_2 + " [SEP]"
        emp2_tokens = tokenizer.tokenize(marked_emp2)
        
        if len(emp2_tokens) > 512:
            emp2_tokens =emp2_tokens[0:511] +['SEP']
        
        idx_emp2_tokens = tokenizer.convert_tokens_to_ids(emp2_tokens)
        segments_idf_emp2 = [1] * len(emp2_tokens)

        emp2_tokens_tensor = torch.tensor([idx_emp2_tokens]) # type: torch.Tensor
        emp2_segments_tensors = torch.tensor([segments_idf_emp2])
    
    
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    
    try: 
        with torch.no_grad():

            dsp_outputs = model(dsp_tokens_tensor, dsp_segments_tensors)
            dsp_hidden_states = dsp_outputs[2]
        
            edu_outputs = model(edu_tokens_tensor, edu_segments_tensors)
            edu_hidden_states = edu_outputs[2]
        
            emp_outputs = model(emp_tokens_tensor, emp_segments_tensors)
            emp_hidden_states = emp_outputs[2]
            
            jobTitle_outputs = model(jobTitle_tokens_tensor, jobTitle_segments_tensors)
            jobTitle_hidden_states = jobTitle_outputs[2]
            
            
            if c_dsp_separate:     
                dsp2_outputs = model(dsp2_tokens_tensor, dsp2_segments_tensors)
                dsp2_hidden_states = dsp2_outputs[2]
            if c_emp_separate:
                emp2_outputs = model(emp2_tokens_tensor, emp2_segments_tensors)
                emp2_hidden_states = emp2_outputs[2]
    
    except:
        error_idx.append(each_company)
        error = True
        print("error happens here at :" + str(each_company))
        
        pass
    
    
      
    if error == False:
        # `hidden_states` has shape [13 x 1 x #tokens x 768]
        # `token_vecs` is a tensor with shape [#tokens x 768]
        # Calculate the average of all 22 token vectors.
        
        if c_dsp_separate:
            dsp1_embedding = get_SentenceVector(dsp_hidden_states)
            dsp2_embedding = get_SentenceVector(dsp2_hidden_states)
            dsp_embedding = dsp1_embedding + dsp2_embedding
        else:
            dsp_embedding = get_SentenceVector(dsp_hidden_states)
        
        
        
        if c_emp_separate:
            emp1_embedding = get_SentenceVector(emp_hidden_states)
            emp2_embedding = get_SentenceVector(emp2_hidden_states)
            emp_embedding = emp1_embedding + emp2_embedding
        else:
            emp_embedding = get_SentenceVector(emp_hidden_states)
        
        edu_embedding = get_SentenceVector(edu_hidden_states)
        jobTitle_embedding = get_SentenceVector(jobTitle_hidden_states)
        
        # Organize the output
        dsp_embedding_list.append(dsp_embedding.tolist())
        edu_embedding_list.append(edu_embedding.tolist())
        emp_embedding_list.append(emp_embedding.tolist())
        jobTitle_embedding_list.append(jobTitle_embedding.tolist())
        
        # Sum all the embedding vector
        total_embedding = dsp_embedding + emp_embedding + edu_embedding + jobTitle_embedding
        total_embedding_list.append(total_embedding.tolist())
    
    else:
        dsp_embedding_list.append(dsp_embedding)
        edu_embedding_list.append(edu_embedding)
        emp_embedding_list.append(emp_embedding)
        jobTitle_embedding_list.append(jobTitle_embedding)
        total_embedding_list.append(total_embedding)
    
    
    
    # ## Organize the output
    # dsp_hiddenstates_list.append(dsp_hidden_states)
    # edu_hiddenstates_list.append(edu_hidden_states)
    # emp_hiddenstates_list.append(emp_hidden_states)
    # jobTitle_hiddenstates_list.append(jobTitle_hidden_states)


    if each_company % 10 == 0:
        print(each_company)



print("Outside the loop, saving data")


#Organize the data into a dictionary  
f_ID_list = f_ID.tolist()
f_org_list = f_orgName.tolist()  
#f1 = f_ID_list[0:10]
#f2 = f_org_list[0:10]
output_dict_1 = {'org_uuid':f_ID_list, 'org_name':f_org_list ,'Descroiption':dsp_embedding_list,'Founder_education':edu_embedding_list,'Founder_employerCompany':emp_embedding_list, 'Founder_jobTitle':jobTitle_embedding_list, }
output_dict_2 = {'org_uuid':f_ID_list, 'org_name':f_org_list , 'Total features' : total_embedding_list}
#output data frame
final_df_1 = pd.DataFrame(output_dict_1)
final_df_2 = pd.DataFrame(output_dict_2)
#Save the dataframe to csv
final_df_1.to_csv('Fail_Founder_Inf_vector.csv')
final_df_2.to_csv('Fail_Founder_sumVector.csv')


print("Finish~")




