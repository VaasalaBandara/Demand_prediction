#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LogisticRegression
import pandas as pd


# In[3]:


# load data from a CSV file
data = pd.read_csv('pricepred5.csv')
print(f"the imported dataframe is\n\n\n{data}")


# In[4]:


#obtaining top 20 rows of csv
subset = data.loc[0:19, ['month', 'inflation','production price','Demand']]
print("\n\n\n")
print(f"the top 10 rows of the dataframe are:\n\n\n{subset}")


# In[5]:


# splitting the data into features (X) and labels (y)
X = subset[['inflation', 'production price']] # select columns for features
y = subset['Demand'] # select column for labels


# In[6]:


# creating a logistic regression model and fit it to the data
clf = LogisticRegression(max_iter=3000) #clf is now the model name
clf.fit(X, y)


# In[7]:


# making predictions on new data
new_data = pd.read_csv('new_data2.csv')
print("\n\n\n")
print(f"the data set where the demand to be predicted is:\n\n\n{new_data}")
X_new = new_data[['inflation', 'production price']]
y_pred = clf.predict(X_new)


# In[8]:


#printing the predicted values
print(y_pred)


# In[9]:


# concatenate the array and dataframe along axis 1 (columns)
combined = pd.concat([new_data, pd.DataFrame(y_pred.T)], axis=1)
combined


# In[10]:


#the values to be predicted
y_new=data.loc[20:39, ['month', 'inflation','production price','Demand']]
print(f"the part of the data set that was predicted:\n\n{y_new}")
#seperating the column of the values to be predicted
y_topredict=y_new["Demand"]
print(f"the column to be predicted is:\n\n\n{y_topredict}")


# In[11]:


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_topredict,y_pred)


# In[ ]:




