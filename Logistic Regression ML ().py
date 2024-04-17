#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("car_data.csv")


# In[61]:


df.head()


# In[62]:


df.info()


# In[63]:


df.describe()


# In[ ]:





# In[64]:


x = df.iloc[:,[2,3]].values
y= df.iloc[:,4].values


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state = 0)


# In[66]:


x_train


# In[67]:


y_train


# Transforming the data into machine for training and testing

# In[68]:


from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()

x_train =st_x.fit_transform(x_train)
x_test =st_x.transform(x_test)


# In[74]:


# used data for training and testing

display('Training Data',x_train)
display('Testing data',x_test)


# # Fitting the LR  for training
# 

# In[22]:


from sklearn.linear_model import LogisticRegression
clasifier = LogisticRegression(random_state=0)
clasifier.fit(x_train,y_train)


# In[23]:


y_pred =clasifier.predict(x_test)


# In[46]:


y_pred


# In[47]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# # Visualization of Training data set

# In[77]:


from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'black'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('AnnualSalary')
plt.legend()
plt.show()


# # Visualization of predicted data set

# In[55]:


from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'yellow'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('AnnualSalary')
plt.legend()
plt.show()


# In[ ]:




