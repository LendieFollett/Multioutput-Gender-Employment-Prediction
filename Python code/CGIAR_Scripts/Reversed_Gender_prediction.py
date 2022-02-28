#!/usr/bin/env python
# coding: utf-8

# In[129]:


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


# In[130]:


# specify the column name and mapping for the labels we are trying to predict
col = 'Q6. Thinking about the last 12 months, how much input did you have in decisions about how to use total HOUSEHOLD income?'
mapping =  ({1: 0,# Respondent
             2: 1,# Spouse
             3: 0,# Respondent and spouse
             4: 1,# Other
            })
gender_col = 'Q1. How should I greet you?'
gender_mapping =  ({1: 1,# Women
             2: 0,# Men 
            })


# In[142]:


def create_train_test(column_gender, col, mapping, gender_mapping, rural =1, region = None):
    
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
    
   
    df = df.merge(targets, left_index=True, right_index=True, how="inner")
    
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
        data = data[data['URBAN']==1]
    elif rural == 0:
        data = data[data['URBAN']==0]
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
        
    
        
   
    
    
     # drop irrelevant columns and merge with labels
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


# In[143]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural =2, region= None)
X_train.shape


# In[144]:


def get_hyperparams(pickle_file): 
    trials = pickle.load(open(pickle_file, "rb"))
    
    results = trials.trials
    
    results = sorted(results, key=lambda x: -x['result']['test_accuracy'])

    if results:
        
        return results[0]['result']['hyper']


# In[145]:


file_path = "../decisions_results_20191114-140555.pickle"
params =  get_hyperparams(file_path)
params


# In[146]:


def Decision_Model(X_train, y_train, params):
        
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model
Decision_model = Decision_Model(X_train, y_train, params)

dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = "../Models/decisions_model.pickle"
pickle.dump(Decision_model, open(model_path, 'wb'))
Decision_model


# In[ ]:





# In[147]:


def predict(model, X_test, y_test):
    
    y_test = np.array(y_test).astype(int)
   
    prediction = model.predict(X_test)
   
    test_df = pd.DataFrame(X_test)

    test_accuracy = accuracy_score(y_test, prediction)
    return test_accuracy, test_df, prediction


test_accuracy, test_df, prediction = predict(Decision_model, X_test, y_test)
test_accuracy


# In[148]:


# select gennder to make land prediction on
def Select_Decision(decision,test_df):
    data_df = test_df
    data_df['ID'] = y_test.index
    data_df['ground_truth_decision'] = y_test.values
    data_df['predicted_decision'] = prediction 
    data_df['gender_y_test'] = y_test_gender.values
    data_df.set_index('ID', inplace =True, drop = True)
    
    if decision == 1:
        data = data_df[data_df['predicted_decision']==1]
    elif decision ==0:
        data = data_df[data_df['predicted_decision']==0]
    else:
        data = data_df[data_df['predicted_decision']==2]
        
    land_X_test = data.drop(['ground_truth_decision','predicted_decision','gender_y_test'], axis=1)
    land_y_test = data['gender_y_test'].copy()
    
    return [land_X_test , land_y_test] 


# In[ ]:





# In[149]:


file_path = "../gender_results_20191114-140400.pickle"
params =  get_hyperparams(file_path)
params


# In[150]:


def Train_gender(X_train, y_train, params):
    
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model
Gender_model = Train_gender(X_train, y_train_gender, params)
Gender_model

dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = "../Models/Gender_model.pickle"
pickle.dump(Gender_model, open(model_path, 'wb'))
Gender_model


# In[163]:


def confusion_matrix(y_test, prediction):
    from sklearn import metrics

    confusion = metrics.confusion_matrix(y_test, prediction)
    print(confusion)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    #FP1 = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    #FN1 = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    #TP1 = np.diag(cnf_matrix)
    #TN1 = cnf_matrix.sum() - (FP + FN + TP)

    FP1 = FP.astype(float)
    FN1 = FN.astype(float)
    TP1 = TP.astype(float)
    TN1 = TN.astype(float)
    
   # print((TP + TN) / float(TP + TN + FP + FN))
    #print(metrics.accuracy_score(y_test, prediction))
    
    #classification_error = (FP + FN) / float(TP + TN + FP + FN)

    #print(classification_error)
    #print(1 - metrics.accuracy_score(y_test, prediction))
    
    sensitivity = TP / (FN + TP)

    print(sensitivity)
    #print(metrics.recall_score(y_test, prediction, average='micro'))
    
    
    specificity = TN / (TN + FP)

    print(specificity)
    
    precision = TP / (TP + FP)

    print(precision)
    #print(metrics.precision_score(y_test, prediction,average='micro'))


# In[164]:


## Predict female on the yes decision
X, y = Select_Decision(1,test_df)

test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[166]:


# Predict Decision power for combined male and female
X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# # House 

# In[168]:


# specify the column name and mapping for the labels we are trying to predict
house_col = 'Q7. Do you own your own house or agricultural land, either solely or jointly with someone else?'
house_mapping =  ({1: 0,# Land
            2: 1,# House 
            3: 1,# Both
            4: 0,# Neither
            })


# In[169]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=house_col, mapping = house_mapping, gender_mapping=gender_mapping, rural =2, region= None)


# In[170]:


file_path = "../house_results_20200124-191817.pickle"
params =  get_hyperparams(file_path)
params


# In[171]:


def House_Model(X_train, y_train, params):
    
    params['degree'] = int(params['degree'])
   
    y_train = y_train.astype(int)
    
    model = SVC(**params)
   
    model.fit(X_train, y_train)
    
    return model
House_model = House_Model(X_train, y_train, params)

dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = "../Models/house_model_{}.pickle".format(dtime)
pickle.dump(House_model, open(model_path, 'wb'))
House_model


# In[172]:


test_accuracy, test_df, prediction = predict(House_model, X_test, y_test)
test_accuracy


# In[173]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# In[174]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# ### Land 

# In[175]:


# specify the column name and mapping for the labels we are trying to predict
col = 'Q7. Do you own your own house or agricultural land, either solely or jointly with someone else?'
mapping =  ({1: 1,# Land
             2: 0,# House 
             3: 1,# Both
             4: 0,# Neither
            })


# In[176]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural =2, region= None)


# In[177]:



file_path = "../land_results_20191115-080146.pickle"
params =  get_hyperparams(file_path)
params


# In[178]:


def Land_Model(X_train, y_train, params):
     
    params['degree'] = int(params['degree'])
   
    
    y_train = y_train.astype(int)
    
    model = SVC(**params)
    
    model.fit(X_train, y_train)
    
    return model
Land_model = Land_Model(X_train, y_train, params)
dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = "../Models/Land_model_{}.pickle".format(dtime)
pickle.dump(Land_model, open(model_path, 'wb'))
Land_model


# In[179]:


test_accuracy, test_df, prediction = predict(Land_model, X_test, y_test)
test_accuracy


# In[182]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[183]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# # Occupation

# ### (3 Agric,  Non agric, no labour force)

# In[184]:


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


# In[185]:


X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural =2, region= None)


# In[186]:


file_path = "../occupations_agri_results_20191117-103453.pickle"
params =  get_hyperparams(file_path)
params


# In[188]:


def Agric_Non_Agric_Model(X_train, y_train, params):
     
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model
Agric_Non_Agric_model = Agric_Non_Agric_Model(X_train, y_train, params)
dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = "../Models/Agric_Non_Agric_Model{}.pickle".format(dtime)
pickle.dump(Agric_Non_Agric_model, open(model_path, 'wb'))
Agric_Non_Agric_model


# In[189]:


test_accuracy, test_df, prediction = predict(Agric_Non_Agric_model, X_test, y_test)
test_accuracy


# In[193]:


np.unique(prediction)


# In[198]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values ,y)

test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# In[199]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)

test_accuracy
confusion_matrix(y, prediction1)


# In[200]:


X, y = Select_Decision(2,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)

test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# ## (3: Without wage, expected wage, out of the workforce)

# In[201]:


# specify the column name and mapping for the labels we are trying to predict
col = 'Q5. Can you please tell me your main occupation (the one you get most of your income from) in the last 12 months?'
mapping =  ({1: 1,# Subsistence farmer
            2: 0,# Commercial farmer 
            3: 0,# Agricultural wage worker
            4: 0,# Non agriculture self-employed
            5: 0,# Non agriculture wage-workers with contract
            6: 0,# Non agriculture wage-workers without a contract 
            7: 2,# Retired or unemployed and not looking for work
            8: 2,# Student 
            9: 2,# Housewife
            10: 2,# Unemployed  
           })
X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2)


# In[202]:


file_path = "../occupations_agri_results_20191117-103453.pickle"
params =  get_hyperparams(file_path)
def Multi_Model(X_train, y_train, params):
        
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model
Multi_model = Multi_Model(X_train, y_train, params)
model_path = "../Models/Reversed_Multi_model.pickle"
pickle.dump(Multi_model, open(model_path, 'wb'))


# In[117]:


test_accuracy, test_df, prediction = predict(Multi_model, X_test, y_test)
test_accuracy


# In[203]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model,  X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[204]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[205]:


X, y = Select_Decision(2,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# In[ ]:





# ### Farmers, wage earners, Non income(students and retired excluded)

# In[207]:


#3.Farmers, wage earners, Non income(students and retired excluded)

#______________________________________________________________________________


# specify the column name and mapping for the labels we are trying to predict
col = 'Q5. Can you please tell me your main occupation (the one you get most of your income from) in the last 12 months?'
mapping =  ({1: 0,# Subsistence farmer
            2: 0,# Commercial farmer 
            3: 1,# Agricultural wage worker
            4: 1,# Non agriculture self-employed
            5: 1,# Non agriculture wage-workers with contract
            6: 1,# Non agriculture wage-workers without a contract 
            7: None,# Retired or unemployed and not looking for work
            8: None,# Student 
            9: 2,# Housewife
            10: 2,# Unemployed  
           })
X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2)


# In[208]:


file_path = "../occupations_marcelo_results_20191117-102923.pickle"
params =  get_hyperparams(file_path)

def Marcelo_Model(X_train, y_train, params):
    
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model
Marcelo_model = Marcelo_Model(X_train, y_train, params)
model_path = "../Models/Reversed_Marcelo_model.pickle"
pickle.dump(Marcelo_model, open(model_path, 'wb'))
Marcelo_model

test_accuracy, test_df, prediction = predict(Marcelo_model, X_test, y_test)
test_accuracy


# In[209]:


test_accuracy, test_df, prediction = predict(Marcelo_model, X_test, y_test)
test_accuracy


# In[215]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model ,X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[216]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[217]:


X, y = Select_Decision(2,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy
confusion_matrix(y, prediction1)


# In[ ]:





# ###  Wage or no wage

# In[ ]:


# specify the column name and mapping for the labels we are trying to predict
col = 'Q5. Can you please tell me your main occupation (the one you get most of your income from) in the last 12 months?'
mapping =  ({1: 1,# Subsistence farmer
            2: 0,# Commercial farmer 
            3: None,# Agricultural wage worker
            4: 0,# Non agriculture self-employed
            5: 0,# Non agriculture wage-workers with contract
            6: 0,# Non agriculture wage-workers without a contract 
            7: None,# Retired or unemployed and not looking for work
            8: None,# Student 
            9: 1,# Housewife
            10: 1,# Unemployed  
           })
X_train, y_train, X_test, y_test, y_train_gender, y_test_gender = create_train_test(column_gender=gender_col, col=col, mapping = mapping, gender_mapping=gender_mapping, rural=2)


# In[54]:



file_path = "../decisions_results_20191114-140555.pickle"
params =  get_hyperparams(file_path)
def Wage_or_Model(X_train, y_train, params):
    
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
   
    y_train = y_train.astype(int)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

Wage_or_model = Wage_or_Model(X_train, y_train, params)
model_path = "../Models/Reversed_Wage_or_model.pickle"
pickle.dump(Wage_or_model, open(model_path, 'wb'))
Wage_or_model


# In[127]:


test_accuracy, test_df, prediction = predict(Wage_or_model, X_test, y_test)
test_accuracy


# In[128]:


X, y = Select_Decision(0,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X_test, y_test_gender)
test_accuracy


# In[57]:


X, y = Select_Decision(1,test_df)
test_accuracy, test_df1, prediction1= predict(Gender_model, X.values, y)
test_accuracy


# In[ ]:





# In[ ]:





# In[ ]:




