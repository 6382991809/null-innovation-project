#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train=pd.read_csv ('Tweet_NFT.xlsx - Sheet1.csv')


# In[3]:


train .head()


# In[4]:


train.isnull()


# In[5]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='tweet_intent',data=train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='tweet_intent',hue='tweet_created_at',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='tweet_intent',hue='tweet_text',data=train,palette='rainbow')


# In[ ]:


sns.distplot(train['tweet_intent'].dropna(),kde=False,color='darkred',bins=40)


# In[ ]:


train['tweet_intent'].hist(bins=30,color='darkred',alpha=0.3)


# In[ ]:


sns.countplot(x='id',data=train)


# In[ ]:


train['tweet_text'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['tweet_intent'].iplot(kind='hist',bins=30,color='green')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='tweet_text',y='tweet_intent',data=train,palette='winter')


# In[ ]:


def impute_tweet_intent(cols):
    tweet_intent = cols[0]
    tweet_created_at = cols[1]
    
    if pd.isnull(tweet_intent):

        if tweet_created_at == 2022-08-06T16:56:36.000Z:
            return community

        elif tweet_created_at == 2022-08-06T16:56:35.000Z:
            return appreciation

        else:
            return 24

    else:
        return tweet_intent


# In[ ]:


train['tweet_intent'] = train[['tweet_intent','tweet_created_at']].apply(impute_tweet_intent,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('id',axis=1,inplace=True)


# In[ ]:


train.info()


# In[ ]:


pd.get_dummies(train['tweet_text'],drop_first=True).head()


# In[ ]:


tweet_created_at = pd.get_dummies(train['tweet_created_at'],drop_first=True)
tweet_intent = pd.get_dummies(train['tweet_intent'],drop_first=True)


# In[ ]:


train.drop(['tweet_created_at','tweet_intent','id','tweet_text'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train = pd.concat([train,tweet_created_at,tweet_intent],axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop('tweet_intent',axis=1).head()


# In[ ]:


train['tweet_intent'].head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('tweet_intent',axis=1), 
                                                    train['tweet_intent'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


accuracy=confusion_matrix(y_test,predictions)


# In[ ]:


accuracy


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))

