# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:50:31 2020

@author: dhaar01
"""

import numpy as np
import pandas as pd

#Import sampling helper and preprocessing modules
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#Import random forest model
from sklearn.ensemble import RandomForestRegressor

#Import cross-validation pipeline, evaluation metrics & module for saving models.
#Joblib is an alternative to Python's pickle package
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

#Load wine data from remote URL
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#Split the data into training and test sets
y = data.quality
x = data.drop('quality', axis=1)

#Setting aside 20% of the data as a test set for evaluating the model.
#Also setting an arbitrary "random state" (aka seed) so we can reproduce results
#It's good practice to stratify your sample by the target variable.  This will
#ensure your training set looks similar to your test set, improving reliability of evaluation metrics.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    random_state = 123,
                                                    stratify=y)

#Standardization is the process of subtracting the mean from each feature and divding by stand. devia.

#Transformer API allows you to fit a preprocessing step using the training data the same way
#you'd fit a model and then use the same transformation on future data sets.

#Fitting the Transformer API, applying transformer to training data, and then applying to test data

#scaler = preprocessing.StandardScaler().fit(x_train)
#x_train_scaled = scaler.transform(x_train)

# =============================================================================
# print (x_train_scaled.mean(axis=0))
# print (x_train_scaled.std(axis=0))
# =============================================================================

#x_test_scaled = scaler.transform(x_test)

# =============================================================================
# print (x_test_scaled.mean(axis=0))
# print (x_test_scaled.std(axis=0))
# =============================================================================

#In practice, when we set up the cross-validation pipeline, we won't need to manually fit the
#Transformer API.  Instead we simply declare the class object:
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# =============================================================================
# print(pipeline.get_params())
# =============================================================================

#Declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

#GridSearchCV essentially performs cross-validation across the entire "grid" of hyperparameters

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(x_train, y_train)

#Identify the best set of parameters found using CV
print (clf.best_params_)

#Confirm model will be retrained on entire training set (sklearn does this automaticall)
print (clf.refit)

#Predict a new set of data and use metrics imported earlier to evaluate our model performance
y_pred = clf.predict(x_test)
print (r2_score(y_test, y_pred))
print (mean_squared_error(y_test, y_pred))


# =============================================================================
# Complete Code Below w/ more concise notes
# =============================================================================


# 2. Import libraries and modules
import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
 
# 3. Load red wine data.
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
 
# 4. Split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
 
# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
 
# 6. Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
clf.fit(X_train, y_train)
 
# 8. Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(X_test)
print (r2_score(y_test, pred))
print (mean_squared_error(y_test, pred))
 
# 10. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')
