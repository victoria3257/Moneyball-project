#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:35:01 2022

Use BERT to convert the paragraphs or phrases in each cell into a vector

This code applies to the company list spreadsheet:


Work under the environment -py version 3.8.15

@author: Yuanmin Zhang
"""

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy
import math
import nltk
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
os.chdir('Please enter the path here')
folder_path = os.getcwd()
filename_1= 'Fail_company_Organized_cleanup'  #'Fail_company_Organized_cleanup'

f_company_list = pd.read_csv(folder_path+'/'+filename_1+'.csv')

f_ID = f_company_list['org_uuid']
f_orgName = f_company_list['org_name']
f_category= f_company_list['category_groups_list'] 
f_country = f_company_list['country_code'] 
f_city= f_company_list['city']
f_dsp = f_company_list['description'] 


dsp_hiddenstates_list = []
category_hiddenstates_list = []
country_hiddenstates_list = []
city_hiddenstates_list = []

dsp_embedding_list = []
category_embedding_list = []
country_embedding_list = []
city_embedding_list = []
total_embedding_list = []

error_idx = []
each_company = 0

for each_company in range(len(f_dsp)):
#for each_company in range(3193,len(f_dsp)):
    
    error = False  
    c_dsp_separate = False
    
    dsp_hidden_states = []
    category_hidden_states = []
    country_hidden_states = []
    city_hidden_states = []
    
    dsp_embedding = []
    category_embedding = []
    country_embedding = []
    city_embedding = []
    total_embedding = []
    

    # clean the sentences
    c_dsp = clean_sentences(f_dsp[each_company])
    c_category = deepclean_sentences(f_category[each_company])
    c_country = deepclean_sentences(f_country[each_company])
    c_city = deepclean_sentences(f_city[each_company])
    
    if len(c_dsp) > 2500:
        [c_dsp,c_dsp_2] = separate_paragraphs(c_dsp)
        c_dsp_separate = True

    
    # Adjust the format of sentences for BERT model
    # Mark each sentence with starting [CLC] and end with [SEP]
    marked_dsp = "[CLS] " + c_dsp + " [SEP]"
    marked_category = "[CLS] " + c_category + " [SEP]"
    marked_country = "[CLS] " + c_country + " [SEP]"
    marked_city = "[CLS] " + c_city + " [SEP]"

    dsp_tokens = tokenizer.tokenize(marked_dsp) # dsp_tkn: tokenized description
    category_tokens = tokenizer.tokenize(marked_category)
    country_tokens = tokenizer.tokenize(marked_country)
    city_tokens = tokenizer.tokenize(marked_city)
    
    if len(dsp_tokens) > 512:
        dsp_tokens = dsp_tokens[0:511] + ['SEP']
    
    # Map the token strings to their vocabulary indeces.
    idx_dsp_tokens = tokenizer.convert_tokens_to_ids(dsp_tokens)
    idx_category_tokens = tokenizer.convert_tokens_to_ids(category_tokens)
    idx_country_tokens = tokenizer.convert_tokens_to_ids(country_tokens)
    idx_city_tokens = tokenizer.convert_tokens_to_ids(city_tokens)


    # Mark each of the tokens as belonging to sentence "1".
    # The input is only single sentence
    segments_idf_dsp = [1] * len(dsp_tokens)
    segments_idf_category = [1] * len(category_tokens)
    segments_idf_country = [1] * len(country_tokens)
    segments_idf_city = [1] * len(city_tokens)
    
    # Convert inputs to PyTorch tensors
    dsp_tokens_tensor = torch.tensor([idx_dsp_tokens]) # type: torch.Tensor
    dsp_segments_tensors = torch.tensor([segments_idf_dsp]) # type: torch.Tensor
    category_tokens_tensor = torch.tensor([idx_category_tokens]) # type: torch.Tensor
    category_segments_tensors = torch.tensor([segments_idf_category]) # type: torch.Tensor
    country_tokens_tensor = torch.tensor([idx_country_tokens]) # type: torch.Tensor
    country_segments_tensors = torch.tensor([segments_idf_country]) # type: torch.Tensor
    city_tokens_tensor = torch.tensor([idx_city_tokens]) # type: torch.Tensor
    city_segments_tensors = torch.tensor([segments_idf_city]) # type: torch.Tensor
    
    
    if c_dsp_separate:
        
        marked_dsp2 = "[CLS] " + c_dsp_2 + " [SEP]"
        dsp2_tokens = tokenizer.tokenize(marked_dsp2)
        
        if len(dsp2_tokens) > 512:
            dsp2_tokens = dsp2_tokens[0:511] + ['SEP']
        
        idx_dsp2_tokens = tokenizer.convert_tokens_to_ids(dsp2_tokens)
        segments_idf_dsp2 = [1] * len(dsp2_tokens)

        dsp2_tokens_tensor = torch.tensor([idx_dsp2_tokens]) # type: torch.Tensor
        dsp2_segments_tensors = torch.tensor([segments_idf_dsp2]) # type: torch.Tensor
    
    
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    
    try: 
        with torch.no_grad():

            dsp_outputs = model(dsp_tokens_tensor, dsp_segments_tensors)
            dsp_hidden_states = dsp_outputs[2]
        
            category_outputs = model(category_tokens_tensor, category_segments_tensors)
            category_hidden_states = category_outputs[2]
        
            country_outputs = model(country_tokens_tensor, country_segments_tensors)
            country_hidden_states = country_outputs[2]
            
            city_outputs = model(city_tokens_tensor, city_segments_tensors)
            city_hidden_states = city_outputs[2]
            
            if c_dsp_separate:     
                dsp2_outputs = model(dsp2_tokens_tensor, dsp2_segments_tensors)
                dsp2_hidden_states = dsp2_outputs[2]
    
    except:
        error_idx.append(each_company)
        error = True
        print("errors happen here at :" + str(each_company))
        pass
    
    
    # `hidden_states` has shape [13 x 1 x #tokens x 768]
    # `token_vecs` is a tensor with shape [#tokens x 768]
    # Calculate the average of all 22 token vectors.
    
    if error == False:
        
        
        
        
        # Description contain paragraphs: 
        if c_dsp_separate:
            dsp1_embedding = get_SentenceVector(dsp_hidden_states)
            dsp2_embedding = get_SentenceVector(dsp2_hidden_states)
            dsp_embedding = dsp1_embedding + dsp2_embedding
        else:
            dsp_embedding = get_SentenceVector(dsp_hidden_states)    
        category_embedding = get_SentenceVector(category_hidden_states)
    
        # Country, city and category are words/phrases: 
        country_embedding = get_WordVector(country_hidden_states, c_country, country_tokens)
        city_embedding = get_WordVector(city_hidden_states, c_city, city_tokens)
    
        # Organize the output
        dsp_embedding_list.append(dsp_embedding.tolist())
        category_embedding_list.append(category_embedding.tolist())
        country_embedding_list.append(country_embedding.tolist())
        city_embedding_list.append(city_embedding.tolist())
        
        # Sum all the embedding vector
        total_embedding = dsp_embedding + category_embedding + country_embedding + city_embedding
        total_embedding_list.append(total_embedding.tolist())
        
    else:
        dsp_embedding_list.append(dsp_embedding)
        category_embedding_list.append(category_embedding)
        country_embedding_list.append(country_embedding)
        city_embedding_list.append(city_embedding)
        
        total_embedding_list.append(total_embedding)
    

    # ## Organize the output
    # dsp_hiddenstates_list.append(dsp_hidden_states)
    # category_hiddenstates_list.append(category_hidden_states)
    # country_hiddenstates_list.append(country_hidden_states)
    # city_hiddenstates_list.append(city_hidden_states)


    if each_company % 10 == 0:
        print(each_company)



print("Outside the loop, saving data")

#dsp_embedding_pd = pd.Series(city_embedding_list)
#t4.to_csv('test_File2.csv')

#Organize the data into a dictionary  
f_ID_list = f_ID.tolist()
f_org_list = f_orgName.tolist()  
#f1 = f_ID_list[3193:len(f_dsp)]
#f2 = f_org_list[3193:len(f_dsp)]
output_dict = {'org_uuid':f_ID_list, 'org_name':f_org_list ,'category_groups_list':category_embedding_list,'Country':country_embedding_list, 'City':city_embedding_list, 'Descroiption':dsp_embedding_list}
output_dict_2 = {'org_uuid':f_ID_list, 'org_name':f_org_list , 'Total Company features' : total_embedding_list}
#output_dict = {'org_uuid':f1, 'org_name':f2 ,'category_groups_list':category_embedding_list,'Country':country_embedding_list, 'City':city_embedding_list, 'Descroiption':dsp_embedding_list}

#output data frame
final_df = pd.DataFrame(output_dict)
final_df_2 = pd.DataFrame(output_dict_2)

#Save the dataframe to csv
final_df.to_csv('Fail_company_vectors2.csv')
final_df_2.to_csv('Fail_company_sumVector.csv')


print("Finish~")






