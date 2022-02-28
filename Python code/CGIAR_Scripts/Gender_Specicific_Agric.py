#!/usr/bin/env python
# coding: utf-8

# In[118]:


import datetime
import os
import pickle
import shutil
import time

import hyperopt
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK, STATUS_FAIL
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb


# In[119]:


# specify the column name and mapping for the labels we are trying to predict
col = 'Q5. Can you please tell me your main occupation (the one you get most of your income from) in the last 12 months?'
mapping =  ({1: 1,# Subsistence farmer
            2: 1,# Commercial farmer 
            3: 1,# Agricultural wage worker
            4: 2,# Non agriculture self-employed
            5: 2,# Non agriculture wage-workers with contract
            6: 2,# Non agriculture wage-workers without a contract 
            7: 0,# Retired or unemployed and not looking for work
            8: 0,# Student 
            9: 0,# Housewife
            10: 0,# Unemployed  
           })
gender_col = 'Q1. How should I greet you?'
gender_mapping =  ({1: 1,# Women
             2: 0,# Men 
            })


# In[126]:


def create_train_test(column_gender, col, mapping, gender_mapping, rural=1, region = None):
    
    # load features
    features_path = '../../data/features_20191108.csv'
    df = pd.read_csv(features_path, dtype={'msisdn': str})
    df.set_index('msisdn', inplace=True, drop=True)
    # load labels
    path = '../../data/Gender Survey Data - All Zones 07-11-2019 V4.xlsx'
    targets = pd.read_excel(path, dtype={"MSISDN": str}, skiprows=[1])
    
        
    # disregard unnecessary columns, map values to binary representation, and drop null values
    targets = targets[['MSISDN', col, column_gender ]]
    targets['target'] = targets[col].map(mapping)
    targets['gender'] = targets[column_gender].map(gender_mapping)
    targets = targets[~pd.isna(targets["target"])]
    targets.set_index('MSISDN', inplace=True, drop=True)
    targets.drop(col, axis=1, inplace=True)
    targets.drop(gender_col, axis=1, inplace =True)
    
    
    
     # create new momo columns
    df["momo_p2p_received_balance_dif_avg_neg"] = [1 if x <0 else 0 for x in df["momo_p2p_received_balance_dif_avg"]]
    df["momo_p2p_sent_balance_dif_avg_neg"] = [1 if x <0 else 0 for x in df["momo_p2p_sent_balance_dif_avg"]]
    
    # drop columns that have more than a certain fraction (e.g. 50%) of null values
    threshold = 0.5
    df = df[[x for x in df.columns if (df[x].isna().sum()/df.shape[0] < threshold)]]
    
    # drop irrelevant columns and merge with labels
    #df = df.drop(["SITE_ID", "NAME_1", "NAME_3"], axis=1)
    df = df.merge(targets, left_index=True, right_index=True, how="inner")
    print(df['NAME_2'].unique())
    # one-hot encode second distric names
    one_hot = pd.get_dummies(df['NAME_2'])
    # fill missing values with zeros - empirically better than using mean or median
    df_zero = df.fillna(0)
    
    # merge with one-hot encoded features and drop it from df
    data = df_zero.merge(one_hot, left_index=True, right_index=True)
    data = data.drop(['NAME_2'], axis=1)
    data_df = data.copy()
    data_df = data_df.reset_index() 
   
        
    if rural==1:
        data = data[data['URBAN']==0]
    elif rural == 0:
        data = data[data['URBAN']==1]
    elif rural == 2:
        data = data
        
        
    if region == "Western Uganda":
        
        data = data[data["NAME_1"]=="Western Uganda"]
        
    elif region == "Eastern Uganda":
         data = data[data["NAME_1"]=="Eastern Uganda"]
            
    elif region == "Central Uganda":
        data = data[data["NAME_1"]=="Central Uganda"]
        
    elif region == "Northern Uganda":
        data = data[data["NAME_1"]=="Northern Uganda"]
        
        
        
        
    else:
        data = data  
        
           

        
    print(data.shape)
    
    data = data.drop(["SITE_ID", "NAME_1", "NAME_3"], axis=1)
    #split data into training and test sets, maintaning the same label proportion
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['target']):
        strat_train_set = data.iloc[train_index]
        strat_test_set = data.iloc[test_index]
        
    X_train = strat_train_set.drop(['target','gender'], axis=1)
    y_train = strat_train_set['target'].copy()
    y_train_gender = strat_train_set['gender'].copy()
    X_test = strat_test_set.drop(['target','gender'], axis=1)
    y_test = strat_test_set['target'].copy()
    y_test_gender = strat_test_set['gender'].copy()
    
    # scale features - mostly useful for certain classifiers (e.g. SVMs)
    # take log(1+x) of features, since they mostly follow an exponential distribution
    X_train = np.log(1 + X_train)
    X_test = np.log(1 + X_test)
   
    # scale features so that they have 0 mean and std equal to 1
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    return [X_train, y_train, X_test, y_test, y_train_gender, y_test_gender]


# In[ ]:





# In[127]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2)


# In[128]:


X_test.shape


# In[129]:


X_train.shape


# ### Gender Prediction 
# 

# In[130]:


def get_hyperparams(pickle_file):
   
    trials = pickle.load(open(pickle_file, "rb"))
    
    results = trials.trials
    
    results = sorted(results, key=lambda x: -x['result']['test_accuracy'])

    if results:
        
        return results[0]['result']['hyper']


# In[131]:


file_path = "../gender_results_20191114-140400.pickle"
params =  get_hyperparams(file_path)
params


# In[132]:


def Train_Gender(X_train, y_train, params):
    
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


# In[133]:


Gender_model = Train_Gender(X_train, y_train_gender, params)
Gender_model
model_path = "../Models/Gender_Rural_model.pickle"
pickle.dump(Gender_model, open(model_path, 'wb'))


# In[134]:


def Predict(model, X_test, y_test):
    
    y_test = np.array(y_test).astype(int)
   
    prediction = model.predict(X_test)
   
    test_df = pd.DataFrame(X_test)

    test_accuracy = accuracy_score(y_test, prediction)
    return test_accuracy, test_df, prediction


test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy


# In[135]:


prediction = Gender_model.predict(X)


# In[136]:


prediction.shape


# In[137]:


df = pd.DataFrame(X)
df["gender"] = prediction 


# In[138]:


df.to_csv('features.csv', index=False) 


# ## Gender specific on agri occupation

# In[139]:


# select gennder to make land prediction on
def Select_Gender(gender,test_df):
    data_df = test_df
    data_df['ID'] = y_test_gender.index
    data_df['ground_truth_gender'] = y_test_gender.values
    data_df['predicted_gender'] = prediction 
    data_df['land_y_test'] = y_test.values
    data_df.set_index('ID', inplace =True, drop = True)
    
    if gender == "female":
        data = data_df[data_df['predicted_gender']==1]
    else:
        data = data_df[data_df['predicted_gender']==0]
    land_X_test = data.drop(['ground_truth_gender','predicted_gender','land_y_test'], axis=1)
    land_y_test = data['land_y_test'].copy()
    
    return [land_X_test , land_y_test] 


# In[140]:


#load saved hyperparameters for the best model

file_path = "../occupations_agri_results_20191117-103453.pickle"
params =  get_hyperparams(file_path)
params


# In[141]:


def Agric_Model(X_train, y_train, params):
    

    
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


# In[142]:


Agric_model = Agric_Model(X_train, y_train, params)

model_path = "../Models/Urban_Agric_model.pickle"
pickle.dump(Agric_model, open(model_path, 'wb'))


# In[172]:


def confusion_matrix(y_test, prediction):
    from sklearn import metrics

    cnf_matrix = metrics.confusion_matrix(y_test, prediction)
    print(cnf_matrix)
    #[row, column]
    #TP = confusion[1, 1]
    #TN = confusion[0, 0]
    #FP = confusion[0, 1]
    #FN = confusion[1, 0]
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
   # print((TP + TN) / float(TP + TN + FP + FN))
    #print(metrics.accuracy_score(y_test, prediction))
    
    #classification_error = (FP + FN) / float(TP + TN + FP + FN)

    #print(classification_error)
    #print(1 - metrics.accuracy_score(y_test, prediction))
    
    sensitivity = TP / (FN + TP)

    print(np.mean(sensitivity))
    #print(metrics.recall_score(y_test, prediction, average='micro'))
    
    
    specificity = TN / (TN + FP)

    print(np.mean(specificity))
    
    precision = TP / (TP + FP)

    print(np.mean(precision))
    #print(metrics.precision_score(y_test, prediction,average='micro'))


# In[173]:


# Predict for combined male and female
test_accuracy, test_df1, prediction1= Predict(Agric_model, X_test, y_test)
test_accuracy
confusion_matrix(y_test, prediction1)


# In[145]:


# Predict for combined male and female
test_accuracy, test_df1, prediction1= Predict(Agric_model, X_test, y_test)
test_accuracy
confusion_matrix(y_test, prediction1)


# In[16]:


# Prediction on female
land_X_test, land_y_test = Select_Gender("female", test_df)
test_accuracy, test_df2, prediction2 = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[17]:


# Prediction on male
land_X_test, land_y_test = Select_Gender("male",test_df)
test_accuracy, test_df3, prediction3 = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# ## RURAL - URBAN 

# ### Urban 
# #### Model trained on the entire data 

# In[174]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=1)


# In[175]:


#Predict gender on the urban people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[78]:


# Predict agric occupation on combined gender in urban area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[21]:


#select the female from the urban data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female 
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[22]:


#select the female from the urban data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the female 
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[ ]:





# ## Rural Area

# In[176]:



X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=0)


# In[178]:


#Predict gender on the rural people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[86]:


# Predict agric occupation on combined gender in rural area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[26]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[27]:


#select the male from the rural data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the male in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[ ]:





# # Region 

# ## Western Uganda

# In[179]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2, region = "Western Uganda")


# In[180]:


X_test.shape


# In[181]:


#Predict gender on the urban people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[90]:


# Predict agric occupation on combined gender in urban area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[32]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[33]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# ## Eastern Uganda

# In[183]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2, region = "Eastern Uganda")


# In[184]:


X_test.shape


# In[185]:


#Predict gender on the urban people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[99]:


# Predict agric occupation on combined gender in urban area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[38]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[39]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# ### Central Uganda

# In[186]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2, region = "Central Uganda")


# In[187]:


X_test.shape


# In[188]:


#Predict gender on the urban people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[108]:


# Predict agric occupation on combined gender in urban area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[44]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[45]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# ### Northern Uganda

# In[189]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2, region = "Northern Uganda")


# In[115]:


X_test.shape


# In[191]:


#Predict gender on the urban people
test_accuracy, test_df, prediction = Predict(Gender_model, X_test, y_test_gender)
test_accuracy
confusion_matrix(y_test, prediction)


# In[117]:


# Predict agric occupation on combined gender in urban area
test_accuracy, _ , _ = Predict(Agric_model, X_test, y_test)
test_accuracy


# In[50]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("female", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[51]:


#select the female from the rural data 
land_X_test, land_y_test = Select_Gender("male", test_df)
#predict agric occupation on the female in rural area
test_accuracy, _ , _ = Predict(Agric_model, land_X_test.values, land_y_test)
test_accuracy


# In[ ]:




