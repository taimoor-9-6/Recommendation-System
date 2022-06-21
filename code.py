import requests
import lxml.html
from lxml import objectify
from random import randint
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter


#For blog pages
res_rest = []
for i in range(13):
    print(i)
    url = 'https://online.datasciencedojo.com/blogs/?blogpage='+str(i)
    print(url)
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'lxml')

    urls_temp_rest = []
    urls_rest=[]
    temp_rest=[]
#     x='blogs'
    for h in soup.find_all('a'):
    #     print(h)
        a = h.get('href')
        urls_temp_rest.append(a)
    for i in urls_temp_rest:
        if i != None :  
            if 'blogs' in i:
                if 'blogpage' in i:
                    None
                else:
                    if 'auth' in i:
                        None
                    else:
                        urls_rest.append(i)
    [temp_rest.append(x) for x in urls_rest if x not in temp_rest]
    for i in temp:
        if i=='https://online.datasciencedojo.com/blogs/':
            None
        else:
            res_rest.append(i)
    print(res_rest)
    print('--------')
    
    
    
#Getting name and description
name=[]
des_temp=[]
for j in res_rest:
    url = j
    response = requests.get(url)
    soup = BeautifulSoup(response.text)

    metas = soup.find_all('meta')
    name.append([ meta.attrs['content'] for meta in metas if 'property' in meta.attrs and meta.attrs['property'] == 'og:title' ])
    des_temp.append([ meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description' ])
    
    
#Removing stop words
stop_words = set(stopwords.words("english"))
descrip=[]
# des_temp=np.array(des_temp)
# print(des_temp)
for i in descrip_temp:
    for j in i:
        text = re.sub("@\S+", "", j)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub("\$", "", text)
        text = re.sub("@\S+", "", text)
        text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])    
    descrip.append(text)

    
    #Building BOW
model = Tokenizer()
model.fit_on_texts(descrip)
rep = model.texts_to_matrix(descrip, mode='count')
# print(f'Key : {list(model.word_index.keys())}')
rep_name=f'Key : {list(model.word_index.keys())}'


#Creating df
# name=np.array(name)
# name=name.astype(str)
df_name=pd.DataFrame(name)
df_name.rename(columns = {0:'name'}, inplace = True)
df_count=pd.DataFrame(rep)
frames=[df_name,df_count]
result=pd.concat(frames,axis=1)
result=result.set_index('name')
result=result.drop([0], axis=1)
for i in range(len(rep)):
    result.rename(columns = {i+1:i}, inplace = True)

    
#Calculating cosine similarity
df_name=df_name.convert_dtypes(str)
a=df_name['name']
sim_df = pd.DataFrame(cosine_similarity(result, dense_output=True))
for i in range(len(name)):
    sim_df.rename(columns = {i:a[i]},index={i:a[i]}, inplace = True)
sim_df


max_val = sim_df.apply(lambda x: pd.Series(np.concatenate([x.nlargest(11).index.values])), axis=1)
max_val
max_val.to_csv('E:/DSD/Internal Analytics/Blogs/Suggested Blogs.csv',index=True)
