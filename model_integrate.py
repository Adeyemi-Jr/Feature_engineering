import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from my_functions import subtract_baseline_glucose
from my_functions import plot_glucose_concentration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

integrated_df = pd.read_csv('../data/processed/20220211/processed_integrated.csv')


#encode the label column
integrated_df['glucose_level'][integrated_df['glucose_level'] == 300] = 0
integrated_df['glucose_level'][integrated_df['glucose_level'] == 1100] = 1
integrated_df['glucose_level'][integrated_df['glucose_level'] == 5000] = 2


y = integrated_df['glucose_level']
X = integrated_df
X.drop(['glucose_level'], axis = 1, inplace = True)
#X.drop(['glucose_level'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


model = RandomForestClassifier(random_state=42)
# model = SVC(random_state=42)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
acc = accuracy_score(y_test, yhat)
print('Accuracy: ', acc)