import pandas as pd 
import numpy as np 
import joblib 
from src.settings import MODELS_DIR
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import os 

class Test:
    """ Test class
    attributes:
    -----------
    df_test_features: test features 
    df_test_label: test_label 

    methods:
    --------
    predict: load model, get predicted data et return performances 
    """

    def __init__(self, df_test_features, df_test_label):
        self.df_test_features = df_test_features
        self.df_test_label = df_test_label 

    def predict(self):
        # load model 
        model = joblib.load(
            os.path.join(
                MODELS_DIR, 
                'ML.pkl'
            )
        )
        predicted = model.predict(self.df_test_features)
        return {
            'accuracy': accuracy_score(predicted, self.df_test_label),
            'roc': roc_auc_score(predicted, self.df_test_label),
            'cm': confusion_matrix(predicted, self.df_test_label, normalize="true")

        }

