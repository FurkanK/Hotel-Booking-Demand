#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv


# load data:
file_path = "C:\\Users\\Furkann\\.spyder-py3\\hotel_bookings.csv"
df = pd.read_csv(file_path)


# In[83]:


df.head()


# In[84]:


df.isnull().sum()


# In[85]:


df.dropna(subset=["children"], axis=0, inplace=True)
df.drop(['country'],axis=1,inplace=True)
zero_guests = list(df.loc[df["adults"]
                   + df["children"]
                   + df["babies"]==0].index)
if zero_guests is not None:
    df.drop(df.index[zero_guests], inplace=True)


# In[86]:


df.drop(['arrival_date_week_number','meal','agent','company','adr','reservation_status','reservation_status_date'],axis=1,inplace=True)


# In[87]:


df = df.reset_index(drop=True)
df.shape


# In[88]:


month_dictionary={'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12 }
df['arrival_date_month']=df['arrival_date_month'].map(month_dictionary)
df.head()


# In[89]:


df = pd.get_dummies(df,columns = ['hotel','market_segment', 'distribution_channel', 'reserved_room_type','assigned_room_type','deposit_type','customer_type'],drop_first=True)
df.head()


# In[90]:


df.describe()


# In[91]:


from sklearn.model_selection import train_test_split

X = df.drop('is_canceled', axis=1) # data
y = df.is_canceled # labels


# In[92]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=0)


# In[93]:


def fit_ml_algo(algo, X_train, y_train, cv):
    
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# In[94]:


# Logistic Regression
train_pred_log, accuracy_log, accuracy_cv_log = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               y_train, 
                                                                    10)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)


# In[98]:


# k-Nearest Neighbours
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  y_train, 
                                                  10)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)


# In[96]:


# Linear SVC
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)


# In[95]:


# Decision Tree Classifier
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)


# In[97]:


# Random Forest Classifier
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(RandomForestClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)


# In[ ]:




