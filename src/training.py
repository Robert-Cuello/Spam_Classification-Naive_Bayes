import os
import re
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preparing_data

def train_models(X_train, y_train, personalised_stop_words):
    ## Training the different models
    pipe = Pipeline([
        
        ('vec', CountVectorizer()),
        ('model', MultinomialNB())
    ])

    vectorizer = [
        CountVectorizer(stop_words=personalised_stop_words, token_pattern=r"(?u)\b\w+\b|[$!?]"),
        TfidfVectorizer(stop_words=personalised_stop_words, token_pattern=r"(?u)\b\w+\b|[$!?]")
    ]

    models = {
        'multinomial_nb': {
            'vec': vectorizer,
            'vec__ngram_range': [(1, 1), (1, 2)],
            'model': [MultinomialNB()],
            'model__alpha': [0.01, 0.1, 0.5, 1.0],
            'model__fit_prior': [True, False]
        },
        'svc': {
            'vec': vectorizer,
            'vec__ngram_range': [(1, 1), (1, 2)],
            'model': [SVC(random_state=42)],
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['linear', 'rbf'],
            'model__class_weight': [None, 'balanced']
        },
        'random_forest': {
            'vec': vectorizer,
            'vec__ngram_range': [(1, 1), (1, 2)],
            'model': [RandomForestClassifier(random_state=42)],
            'model__n_estimators': [100, 200],
            'model__max_depth': [20, 50, None],
            'model__min_samples_split': [2, 5],
            'model__class_weight': [None, 'balanced']
        }
    }

    best_models = {}

    for name, params in models.items():
        grid = GridSearchCV(estimator=pipe, param_grid=params, cv=5, scoring='f1', n_jobs=6)
        grid.fit(X_train, y_train)
        
        best_models[name] = {
            'estimator': grid.best_estimator_,
            'best_params': grid.best_params_,
            'cv_score': grid.best_score_
        }

        print(f'Completed training of model: {name}')

    joblib.dump(best_models, './models/best_models.joblib')

    return best_models

