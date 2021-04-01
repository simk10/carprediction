#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[7]:


df=pd.read_csv("car data.csv")
df.head()


# In[8]:


print(df['Seller_Type'].unique())


# In[9]:


df.isnull().sum()


# In[10]:


newdf=df.drop(['Car_Name'],axis='columns')

newdf['current_year']=2020
newdf['no year']=newdf['current_year']-newdf['Year']

newdf.head()


# In[11]:


newdf=newdf.drop(['Year'],axis='columns')
newdf=newdf.drop(['current_year'],axis='columns')
newdf=pd.get_dummies(newdf,drop_first=True)
newdf.head()
newdf.corr()


# In[12]:


import seaborn as sns
sns.pairplot(newdf)


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=newdf.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(newdf[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[14]:


print(newdf.shape)
X=newdf.iloc[:,1:]
print(X.shape)
y=newdf.iloc[:,0]
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#newdf.shape
#y.head()


# In[15]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]


# In[12]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]


# In[13]:


import sklearn
print(sklearn.__version__)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

randomgrid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}
print(randomgrid)


# In[14]:


rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf, param_distributions=randomgrid,n_iter=10, scoring='neg_mean_squared_error', n_jobs=None)


# In[17]:


print(X.shape)
print('uy')
y.shape
rf_random.fit(X_train,y_train)


# In[20]:


predictions=rf_random.predict(X_test)
predictions


# In[19]:


sns.displot(y_test-predictions)


# In[24]:


from matplotlib.pyplot import plot
plt.scatter(y_test,predictions)


# In[16]:


import pickle
# Save the Modle to file in the current working directory

Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model


# In[ ]:




