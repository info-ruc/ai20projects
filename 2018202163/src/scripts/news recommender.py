# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:17:18 2020

@author: 雷雨寒
"""
import pandas as pd
import requests
import bs4
import numpy as np
from tfidf.tfidf_utils import TfidfRecommender

class reader:
    def __init__(self,uid,like):
        self.uid=uid
        self.like=[]
        self.dislike=[]
        self.like.extend(like)
        
    def read(self,news_id):
        self.like.append(news_id)
    
    def skip(self,news_id):
        self.dislike.append(news_id)

def srisnull(news):
    for i in range(len(news)):
        if(any(news['Abstract'].loc[[i]].isnull())):
            news.drop(news.loc[[i]])

def remove_duplicates(df, cols):
    """ Remove duplicated entries.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list of str): Name of columns in which to look for duplicates.
    
    Returns:
        df (pd.DataFrame): Pandas dataframe with duplicate rows dropped.
    
    """
    for col in cols:
        # Reset index
        df = df.reset_index(drop=True)

        # Find where the identifier variable is duplicated
        dup_rows = np.where(df.duplicated([col]) == True)[0]

        # Drop duplicated rows
        df = df.drop(dup_rows)

    return df


def remove_nan(df, cols):
    """ Remove rows with NaN values in specified column.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list of str): Name of columns in which to look for NaN.
    
    Returns:
        df (pd.DataFrame): Pandas dataframe with invalid rows dropped.
    
    """
    for col in cols:
        # Convert any empty string cells to nan
        df[col].replace("", np.nan, inplace=True)

        # Remove NaN rows
        df = df[df[col].notna()]

    return df


def clean_dataframe(df):
    """ Clean up the dataframe.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
    
    Returns:
        df (pd.DataFrame): Cleaned pandas dataframe.
    """

    # Remove duplicated rows
    cols = ["News ID"]
    df = remove_duplicates(df, cols)

    # Remove rows without values in specified columns
    cols = ['Category','SubCategory','Title','Abstract','URL','Title Entities','Abstract Entities']
    df = remove_nan(df, cols)

    return df

def user_recommendation(recommender,news,reader):
    #data preparation
    news=clean_dataframe(news)
    news=news.reset_index()[0:1000]
    #abstract=news['Abstract']
    
    #print(news)
    
    #fit in tfidf recommender
    tf,vectors_tokenized=recommender.tokenize_text(news,text_col='Abstract')
    
    recommender.fit(tf,vectors_tokenized)
    
    topk=recommender.recommend_top_k_items(news,k=5)
    
    cols_to_keep=['News ID','Title','Abstract']
    
    #print(news.columns)
    
    #generate result
    rec=pd.DataFrame()
    like1=reader.like
    for i in range(len(like1)):
        rec=rec.append(recommender.get_top_k_recommendations(news,like1[i],cols_to_keep))
    
    rec=rec.sort_values(by=['similarity_score'],ascending=False)
    rec['rank']=range(20)
    rec=rec.reset_index()
    rec=rec.drop(columns='index')
    
    #output
    print(uid)
    print(rec[0:10])
    
def create_dictionary(news):
    news_dict={}
    for newsitem in news:
        if(newsitem[Category] not in news_dict.keys()):
            news_dict[newsitem[Category]] = {}
        else:
            dic = news_dict[newsitem['Category']]
            dic[newsitem['SubCategory']].append(newsitem['News ID'])
    print(news_dict)

if __name__=="__main__":
    #load news
    news=pd.read_csv('../dataset/MINDsmall_train/news.tsv',sep='\t',names=['News ID','Category','SubCategory','Title','Abstract','URL','Title Entities','Abstract Entities'])
    news=news.dropna(axis=0,how='any')

    create_dictionary(news)
    #load recommender
    recommender = TfidfRecommender(id_col='News ID', tokenization_method='scibert')
    
    #load behavior
    behaviors=pd.read_csv('../dataset/MINDsmall_train/behaviors.tsv',sep='\t',names=['Impression ID','User ID','Time','History','Impressions'])
    behaviors=behaviors.dropna(axis=0,how='any')
    
    #create a reader
    behave=behaviors[behaviors['Impression ID']==1]
    #like1=str(behave['History'].values[0]).split(' ')
    like1=['N55528','N19639','N61837','N53526']
    uid=behave['User ID'].values
    reader1=reader(uid,like1)
    
    #apply
    user_recommendation(recommender,news,reader1)
    
    

