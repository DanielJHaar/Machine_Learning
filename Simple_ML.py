# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:09:44 2020

@author: dhaar01
"""

#Generate the Training Set
from random import randint
train_set_limit = 1000
train_set_count = 100

train_input = list()
train_output = list()
for i in range(train_set_count):
    a = randint(0, train_set_limit)
    b = randint(0, train_set_limit)
    c = randint(0, train_set_limit)
    op = a + 2*b + 3*c
    train_input.append([a, b, c])
    train_output.append(op)
    
#Train the Model
from sklearn.linear_model import LinearRegression
x = train_input
y = train_output

predictor = LinearRegression(n_jobs=-1)
predictor.fit(x, y)

#Test Data
x_test = [[10, 20, 30]]
outcome = predictor.predict(x_test)
coefficients = predictor.coef_

print('Outcome: {}\nCoefficients: {}'.format(outcome, coefficients))
