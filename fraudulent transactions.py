#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[13]:


data = pd.read_csv("C:/Users/arjun/Downloads/Fraud.csv")


# In[14]:


data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
data = data.fillna("")


# In[15]:


data = pd.get_dummies(data, columns=['type'])


# In[16]:


X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[17]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[19]:


predection = rfc.predict(X_test)
accuracy = accuracy_score(y_test, predection)
print("Model accuracy:", accuracy)


# In[ ]:




