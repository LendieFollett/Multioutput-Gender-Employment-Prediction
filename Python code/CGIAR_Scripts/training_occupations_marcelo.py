#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


def create_train_test(column_name, mapping):
    
    # load features
    features_path = '../data/features_20191108.csv'
    df = pd.read_csv(features_path, dtype={'msisdn': str})
    df.set_index('msisdn', inplace=True, drop=True)
    # load labels
    path = '../data/Gender Survey Data - All Zones 07-11-2019 V4.xlsx'
    targets = pd.read_excel(path, dtype={"MSISDN": str}, skiprows=[1])
    
    # disregard unnecessary columns, map values to binary representation, and drop null values
    targets = targets[['MSISDN', column_name]]
    targets['target'] = targets[col].map(mapping)
    targets = targets[~pd.isna(targets['target'])]
    targets.set_index('MSISDN', inplace=True, drop=True)
    targets.drop(col, axis=1, inplace=True)
    
    # create new momo columns
    df["momo_p2p_received_balance_dif_avg_neg"] = [1 if x <0 else 0 for x in df["momo_p2p_received_balance_dif_avg"]]
    df["momo_p2p_sent_balance_dif_avg_neg"] = [1 if x <0 else 0 for x in df["momo_p2p_sent_balance_dif_avg"]]
    
    # drop columns that have more than a certain fraction (e.g. 50%) of null values
    threshold = 0.5
    df = df[[x for x in df.columns if (df[x].isna().sum()/df.shape[0] < threshold)]]
    
    # drop irrelevant columns and merge with labels
    df = df.drop(["SITE_ID", "NAME_1", "NAME_3"], axis=1)
    df = df.merge(targets, left_index=True, right_index=True, how="inner")

    # one-hot encode second distric names
    one_hot = pd.get_dummies(df['NAME_2'])
    # fill missing values with zeros - empirically better than using mean or median
    df_zero = df.fillna(0)
    
    # merge with one-hot encoded features and drop it from df
    data = df_zero.merge(one_hot, left_index=True, right_index=True)
    data = data.drop(['NAME_2'], axis=1)

    # split data into training and test sets, maintaning the same label proportion
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['target']):
        strat_train_set = data.iloc[train_index]
        strat_test_set = data.iloc[test_index]

    X_train = strat_train_set.drop(['target'], axis=1)
    y_train = strat_train_set['target'].copy()
    X_test = strat_test_set.drop(['target'], axis=1)
    y_test = strat_test_set['target'].copy()
    
    # scale features - mostly useful for certain classifiers (e.g. SVMs)
    # take log(1+x) of features, since they mostly follow an exponential distribution
    X_train = np.log(1 + X_train)
    X_test = np.log(1 + X_test)
    # scale features so that they have 0 mean and std equal to 1
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return [X_train, y_train, X_test, y_test]


# In[4]:


def hyperparameter_space():
    """Define hyperopt hyperparameter space"""
    
    space = hp.choice('classifier_type', [
    {
        'type': 'xgb',
        'n_estimators': hp.quniform('n_estimators', 100, 600, 1),
        'learning_rate': hp.loguniform('learning_rate', -6, -1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'max_depth': hp.quniform('max_depth', 2, 11, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 3.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 3.0)
    },
    {
        'type': 'svm',
        'C': hp.lognormal('C', -4, 10),
        'gamma': hp.loguniform('gamma', -10, 7),
        'degree': hp.quniform('degree', 2, 6, 1),
        'kernel': hp.choice('kernel', ['rbf'])
    },
    {
        'type': 'knn',
        'n_neighbors': hp.quniform('n_neighbors', 3, 8, 1),
        'p': hp.quniform('p', 1, 3, 1),
        'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
        'weights': hp.choice('weights', ['uniform', 'distance'])
    },
    {
        'type': 'random_forest',
        "min_samples_split": hp.quniform("min_samples_split", 2, 200, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 100, 1),
        "max_leaf_nodes": hp.quniform("max_leaf_nodes", 10, 250, 1),
        "criterion": hp.choice('criterion', ["gini", "entropy"]),
        'n_estimators': hp.quniform("n_estimators_rf", 25, 150, 1),
        'max_depth': hp.quniform("max_depth_rf", 50, 250, 1),
        'max_features': hp.uniform('max_features', 0.10, 1.00)
    },
])
    
    return space


# In[5]:


def get_objective(X_train, y_train, X_test, y_test, random_state=None):

    def objective(space):

        classifier_type = space['type']
        del space['type']

        if classifier_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate'],
                subsample=space['subsample'],
                max_depth=int(space['max_depth']),
                colsample_bytree=space['colsample_bytree'],
                reg_lambda=space['reg_lambda'],
                reg_alpha=space['reg_alpha'],
                n_jobs=2,
                random_state=random_state)
        elif classifier_type == 'svm':
            model = SVC(
                **space,
                random_state=random_state)
        elif classifier_type == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=int(space['n_neighbors']),
                p=int(space['p']),
                algorithm=space['algorithm'],
                weights=space['weights'])
        elif classifier_type == 'random_forest':
            model = RandomForestClassifier(
                min_samples_split=int(space["min_samples_split"]),
                min_samples_leaf=int(space["min_samples_leaf"]),
                max_leaf_nodes=int(space["max_leaf_nodes"]),
                criterion=space["criterion"],
                n_estimators= int(space['n_estimators']),
                max_depth=int(space["max_depth"]),
                max_features=space['max_features'],
                n_jobs=2,
                random_state=random_state)
        
        kf = StratifiedKFold(n_splits=5)
        
        start = time.perf_counter()
        try:
            accuracies = []
            for train_index, test_index in kf.split(X_train, y_train):
                # split the training set into train and val folds
                feat_train, target_train = X_train[train_index], y_train[train_index]
                feat_val, target_val = X_train[test_index], y_train[test_index]
                
                model.fit(feat_train, target_train)
                predictions = model.predict(feat_val)
                
                accuracy = accuracy_score(target_val, predictions)
                accuracies.append(accuracy)
                
        except ValueError:
            elapsed = time.perf_counter() - start
            return {'loss': 0.0,
                    'status': STATUS_FAIL,
                    'validation_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'elapsed': elapsed,
                    'hyper': space}
        elapsed = time.perf_counter() - start
        print(model)

        validation_accuracy = np.mean(accuracies)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)
        print('Accuracy {} {}'.format(validation_accuracy, test_accuracy))

        if np.isnan(validation_accuracy):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        return {'loss': -validation_accuracy,
                'status': status,
                'validation_accuracy': validation_accuracy,
                'test_accuracy': test_accuracy,
                'elapsed': elapsed,
                'hyper': space}
    return objective


# In[6]:


def optimize(objective, space, trials_fname=None, max_evals=5):

    if trials_fname is not None and os.path.exists(trials_fname):
        with open(trials_fname, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        trials = Trials()

    fmin(objective,
         space=space,
         algo=tpe.suggest,
         trials=trials,
         max_evals=max_evals)

    if trials_fname is not None:
        temporary = '{}.temp'.format(trials_fname)
        with open(temporary, 'wb') as trials_file:
            pickle.dump(trials, trials_file)
        shutil.move(temporary, trials_fname)

    return trials


# In[7]:


def summarize_trials(trials):
    results = trials.trials

    results = sorted(results, key=lambda x: -x['result']['validation_accuracy'])

    if results:
        print('Best: {}'.format(results[0]['result']))

    results = sorted(results, key=lambda x: -x['result']['test_accuracy'])

    if results:
        print('Best test accuracy: {}'.format(results[0]['result']))


# In[8]:


def main(max_evals):
    # Fix random_state
    seed = 42
    random_state = np.random.RandomState(seed)

    X_train, y_train, X_test, y_test = create_train_test(column_name=col, mapping=mapping)

    dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = './occupations_marcelo_results_{}.pickle'.format(dtime)
    objective = get_objective(X_train, y_train, X_test, y_test, seed)
    space = hyperparameter_space()

    trials = optimize(objective,
                      space,
                      trials_fname=fname,
                      max_evals=max_evals)

    summarize_trials(trials)
    
    return trials


# In[9]:


trials_final = main(max_evals=400)


# In[ ]:




