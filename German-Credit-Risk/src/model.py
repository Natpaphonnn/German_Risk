# model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class CreditRiskModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        # Example preprocessing steps
        self.data.fillna(0, inplace=True)
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    def predict(self, new_data):
        return self.model.predict(new_data)