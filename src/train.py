from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import joblib
import time
import os
import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from src.settings import MODELS_DIR


class Train:
    """ training class -- performs custom random search after sampling data (imbalanced dataset)
    on hyperparameter on lightgbm algorithm instance, fine tune it and load best parameters in 
    MODEL_DIR folder 

    attributes:
    -----------
    df_train_features: training data 
    df_train_label: training label 
    test_size: default=.25

    methods:
    --------
    learning_rate_decay: learning rate grid function for training
    train: global training process -- will be factorize in future version
    """

    def __init__(self, df_train_features=None , df_train_label=None, test_size=.25):
        self.df_train_features = df_train_features
        self.df_train_label = df_train_label
        self.test_size = test_size


    @staticmethod
    def learning_rate_010_decay_power_0995(current_iter):
        base_learning_rate = 0.1
        lr = base_learning_rate  * np.power(.995, current_iter)
        return lr if lr > 1e-3 else 1e-3

    def train(self):
        fit_params = {
            'early_stopping_rounds': 50, 
            'eval_metric' : 'auc', 
            'eval_names': ['valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}

        param_test = {
            'num_leaves': sp_randint(6, 50), 
            'min_child_samples': sp_randint(100, 500), 
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': sp_uniform(loc=0.2, scale=0.8), 
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.df_train_features,
            self.df_train_label,
            test_size=self.test_size
        )
        # random oversampling
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)
        clf = lgb.LGBMClassifier(
            max_depth=-1, 
            random_state=314, 
            silent=True, 
            metric='None', 
            n_jobs=4, 
            n_estimators=900)
        # set randomsearch optimization params
        random_search = RandomizedSearchCV(
            estimator=clf, 
            param_distributions=param_test, 
            scoring='roc_auc',
            cv=5,
            refit=True,
            random_state=314,
            verbose=False)

        fit_params = {
            **fit_params, 
            **{'eval_set': [(X_test, y_test)]}}
        
        # fit random search
        random_search.fit(
            X_train.copy(),
            y_train.copy(), 
            **fit_params)
        
        # build final classifier
        clf_final = lgb.LGBMClassifier(**random_search.best_estimator_.get_params())
        clf_final.fit(
            X_train, 
            y_train, 
            **fit_params, 
            callbacks=[
                lgb.reset_parameter(
                    learning_rate=self.learning_rate_010_decay_power_0995
                    )])
        
        # save model 
        joblib.dump(clf_final, os.path.join(
            MODELS_DIR,
            "{}.pkl".format("ML")))

        predicted = clf_final.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, predicted),
            "roc": roc_auc_score(y_test, predicted),
            "cm": confusion_matrix(y_test, predicted, normalize="true"), 
            "feature_importance": pd.DataFrame(
                sorted(
                    zip(
                        clf_final.feature_importances_,
                        X_test.columns)), 
                    columns=['Value','Feature'])
        }
    


