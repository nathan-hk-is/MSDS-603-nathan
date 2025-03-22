# Nathan Holmes-King
# 2025-03-20
# Lab 2

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Data source:
https://www.kaggle.com/datasets/joebeachcapital/pet-cats-new-zealand

I am trying to predict how many animals a cat catches per month, based on
age and sex.
"""

df = pd.read_csv('../data/cats_NZ/Pet Cats New Zealand-reference-data.csv')
df = df[['animal-life-stage', 'animal-sex', 'animal-comments']]
df['animal-life-stage'] = df['animal-life-stage'].str.split(' ').str[0]
df['animal-sex'] = df['animal-sex'].apply(lambda x: 1 if x == 'm' else 0
                                          if x == 'f' else np.NaN)
df['prey'] = df['animal-comments'].str.split('prey_p_month: ').str[1]
df = df.dropna()
X = df[['animal-life-stage', 'animal-sex']]
y = df['prey']
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('lab-2')

for i in [True, False]:
    mlflow.start_run()
    mlflow.set_tag('model', 'linear regression')
    mlflow.log_param('intercept', i)
    dt = LinearRegression(fit_intercept=i)
    dt.fit(X_train, y_train)
    mlflow.log_metric('train mse',
                      mean_squared_error(y_train, dt.predict(X_train)))
    mlflow.log_metric('test mse',
                      mean_squared_error(y_test, dt.predict(X_test)))
    mlflow.sklearn.log_model(dt, artifact_path='better_models')
    mlflow.end_run()
for i in range(3, 6):
    mlflow.start_run()
    mlflow.set_tag('model', 'decision tree')
    mlflow.log_param('max depth', i)
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(X_train, y_train)
    mlflow.log_metric('train mse',
                      mean_squared_error(y_train, dt.predict(X_train)))
    mlflow.log_metric('test mse',
                      mean_squared_error(y_test, dt.predict(X_test)))
    mlflow.sklearn.log_model(dt, artifact_path='better_models')
    mlflow.end_run()
for i in range(1, 5):
    for j in range(1, 5):
        mlflow.start_run()
        mlflow.set_tag('model', 'random forest')
        mlflow.log_param('n_estimators', i)
        mlflow.log_param('max_features', j)
        dt = RandomForestClassifier(n_estimators=i, max_features=j)
        dt.fit(X_train, y_train)
        mlflow.log_metric('train mse',
                          mean_squared_error(y_train, dt.predict(X_train)))
        mlflow.log_metric('test mse',
                          mean_squared_error(y_test, dt.predict(X_test)))
        mlflow.sklearn.log_model(dt, artifact_path='better_models')
        mlflow.end_run()
