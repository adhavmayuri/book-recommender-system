#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
books=pd.read_csv('BX-Books.csv',sep=";",error_bad_lines=False,encoding='latin-1')


# In[54]:


books


# In[55]:


books.columns


# In[56]:


books=books[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher']]


# In[17]:


books.isna()


# In[18]:


books.rename(columns={'Book-title':'Title','Book-Author':'Author','Year-Of-Publication':'Year'},inplace=True)


# In[19]:


books.head()


# In[8]:


books.shape


# In[60]:


books.head(2)
plt.figure(figsize=(15,7))
sns.countplot(y='Author',data=books,order=pd.value_counts(books['Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[21]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[12]:


users=pd.read_csv('BX-Users.csv',sep=";",error_bad_lines=False,encoding='latin-1')


# In[16]:


users.head(5)


# In[14]:


users.shape


# In[44]:


pip install surprise


# In[64]:


import scipy
import math
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


# In[46]:





# In[65]:


ratings=pd.read_csv('BX-Book-Ratings.csv',sep=";",error_bad_lines=False,encoding='latin-1')


# In[66]:


ratings.inplace=True


# In[67]:


ratings


# In[68]:


user_ratings_threshold = 3

filter_users = ratings['User-ID'].value_counts()
filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()

df_ratings_top = ratings[ratings['User-ID'].isin(filter_users_list)]

print('Filter: users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_ratings_top)))


# In[69]:


book_ratings_threshold_perc = 0.1
book_ratings_threshold = len(df_ratings_top['ISBN'].unique()) * book_ratings_threshold_perc

filter_books_list = df_ratings_top['ISBN'].value_counts().head(int(book_ratings_threshold)).index.to_list()
df_ratings_top = df_ratings_top[df_ratings_top['ISBN'].isin(filter_books_list)]

print('Filter: top %d%% most frequently rated books\nNumber of records: %d' % (book_ratings_threshold_perc*100, len(df_ratings_top)))


# In[70]:


ratings.head(2)


# In[ ]:





# In[71]:


books.shape


# In[72]:


ratings.shape


# In[73]:


users.shape


# In[74]:


ratings.head(2)


# In[75]:


x=ratings['User-ID'].value_counts()>200


# In[76]:


y=x[x].index


# In[77]:


y


# In[78]:


ratings=ratings[ratings['User-ID'].isin(y)]


# In[79]:


ratings.shape


# In[80]:


ratings.head()


# In[81]:


rating_books=ratings.merge(books,on='ISBN')


# In[82]:


rating_books.shape


# In[83]:


number_rating=rating_books.groupby('Book-Title')['Book-Rating'].count().reset_index()


# In[84]:


number_rating.rename(columns={'Book-Rating':'number of rating'},inplace=True)


# In[85]:


number_rating


# In[86]:


final_ratings=rating_books.merge(number_rating,on='Book-Title')


# In[87]:


final_ratings.shape


# In[88]:


final_ratings=final_ratings[final_ratings['number of rating']>=50]


# In[37]:


final_ratings.shape


# In[89]:


final_ratings.drop_duplicates(['Book-Title','User-ID'],inplace=True)


# In[90]:


final_ratings.shape


# In[91]:


book_pivot=final_ratings.pivot_table(columns='User-ID',index='Book-Title',values='Book-Rating')


# In[93]:


book_pivot


# In[94]:


book_pivot.fillna(0,inplace=True)


# In[95]:


book_pivot


# In[96]:


from scipy.sparse import csr_matrix


# In[97]:


book_sparse=csr_matrix(book_pivot)


# In[98]:


type(book_sparse)


# In[99]:


from sklearn.neighbors import NearestNeighbors


# In[100]:


model= NearestNeighbors(algorithm='brute')


# In[101]:


model.fit(book_sparse)


# In[102]:


distances,suggestions=model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1),n_neighbors=6)


# In[103]:


book_pivot


# In[104]:


distances


# In[105]:


suggestions


# In[106]:


for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])


# In[107]:


book_pivot.index[237]


# In[108]:


np.where(book_pivot.index=='Animal Farm')[0][0]


# In[109]:


def recommend_book(Book-Title):
    book_id=np.where(book_pivot.index==Book-Title)[0][0]
    distances,suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)


# In[110]:


for i in range(len(suggestions)):
        if i==0:
              print("The suggestions for",Book-Title,"are:")
        if not i:
      
               print(book_pivot.index[suggestions[i]])


# In[92]:


recommend_book('Animal Farm')


# In[111]:


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)


# In[ ]:




