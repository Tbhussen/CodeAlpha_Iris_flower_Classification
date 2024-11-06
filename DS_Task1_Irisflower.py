#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification Project
# 
# ## Tamim Hussein
# ## Data Science Internship
# 

# In[61]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### Data Extraction

# In[62]:


# loading Iris Flowers Data

df = pd.read_csv('Iris.csv')
df.head()


# ### Data Cleaning

# In[63]:


# remove id colomn because it is not needed

df = df.drop('Id', axis = 1)
df.head()


# In[64]:


# check for Null values

df.isnull().sum()


# ### Data Processing & Visualization

# In[65]:


# count the occurence of each species

df["Species"].value_counts()


# In[66]:


# Describe data

df.describe()


# In[76]:


df.info()


# In[67]:


# Visualize the whole Dataset

sns.pairplot(df, hue = "Species")

plt.show()


# ### Correlation Matrix

# In[68]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

df.corr()


# ### Model Building

# In[69]:


# Separating features and label colomns

data = df.values
# features data
X = data[:,0:4]
# label data
Y = data[:,4]


# In[70]:


# Separating training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


# In[71]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, Y_train)


# In[72]:


model.predict([[6.0, 2.7, 5.1, 1.6]])


# In[73]:


Prediction_test = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("The model accuracy =", accuracy_score(Y_test, Prediction_test) * 100)


# In[74]:


from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(Y_test, Prediction_test)
confusion


# In[75]:


plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Greens', xticklabels=['verginica','versicolor','setosa'], yticklabels=['verginica','versicolor','setosa'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of KNN')
plt.show()

