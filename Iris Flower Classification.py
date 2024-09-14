#!/usr/bin/env python
# coding: utf-8

# 
# # Import Modules

# In[5]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# # Import The Dataset

# In[3]:


iris = pd.read_csv("Iris.csv")


# # Print out the first five rows of the dataset

# In[5]:


print(iris.head())


# In[7]:


#delte a column
iris = iris.drop(columns = ['Id'])
print(iris.head())


# # Descriptive statistics of the dataset

# In[6]:


print(iris.describe())


# In[8]:


iris["Species"].value_counts()


# # Preprocessing the dataset

# In[9]:


iris.isnull().sum()


# # Exploratory Data Analysis
# 

# In[10]:


iris["SepalLengthCm"].hist()


# In[11]:


iris["SepalWidthCm"].hist()


# In[12]:


#Scatterplot
colors = ["red" ,"green","blue"]
species = ["Iris-setosa" , "Iris-versicolor" , "Iris-virginica"]


# In[15]:


for i in range(3):
    x = iris[iris["Species"] == species[i]]
    plt.scatter(x["SepalLengthCm"] , x["SepalWidthCm"] , c=colors[i], label=species[i] )
    
plt.xlabel("Spel Length")
plt.ylabel("Spel Width")
plt.legend()


# In[16]:


for i in range(3):
    x = iris[iris["Species"] == species[i]]
    plt.scatter(x["PetalLengthCm"] , x["PetalWidthCm"] , c=colors[i], label=species[i] )
    
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[17]:


for i in range(3):
    x = iris[iris["Species"] == species[i]]
    plt.scatter(x["SepalLengthCm"] , x["PetalLengthCm"] , c=colors[i], label=species[i] )
    
plt.xlabel("Spel Length")
plt.ylabel("Petal Width")
plt.legend()


# In[18]:


for i in range(3):
    x = iris[iris["Species"] == species[i]]
    plt.scatter(x["SepalWidthCm"] , x["PetalWidthCm"] , c=colors[i], label=species[i] )
    
plt.xlabel("Spel Length")
plt.ylabel("Petal Width")
plt.legend()


# In[35]:


iris.corr()


# # Coorelation Matrix 

# In[36]:


corr = iris.corr()
fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(corr, annot = True , ax = ax)


# # Label Encoding 

# In[1]:


iris = pd.read_csv("Iris.csv")
iris = iris.drop(columns = ['Id'])



# In[10]:


iris.head()


# # Model Training 

# In[2]:


from sklearn.model_selection import train_test_split
X = iris.drop(columns=["Species"])
Y = iris["Species"] 
x_train , x_test, y_train , y_test = train_test_split(X , Y, test_size= 0.2, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


model.fit(x_train , y_train)


# In[61]:


print("Accuracy: ", model.score(x_test , y_test ) * 100)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[ ]:


model.fit(x_train , y_train)


# In[62]:


print("Accuracy: ", model.score(x_test , y_test ) * 100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[ ]:


model.fit(x_train , y_train)


# In[57]:


print("Accuracy: ", model.score(x_test , y_test ) * 100)


# In[5]:


x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(x_new)
print("Prediction: {}".format(prediction))


# In[ ]:





# In[ ]:




