# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:00:01 2022

@author: asus
"""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


dataset = pd.read_csv('F:/bike_rentals.csv')

dataset.head()

dataset=dataset.drop(columns=['dteday','instant'],axis=1)

dataset.isnull().sum()

dataset.info()
dataset.describe()


X= dataset.drop (columns='cnt',axis=1)
y=dataset['cnt']
imputer = SimpleImputer(missing_values = np.nan,
                        strategy ='mean')
SI =imputer.fit(X)
SI=imputer.transform(X)




X_train, X_test, y_train, y_test = train_test_split(SI, y, test_size=0.3, random_state=44, shuffle =True)

scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

baggingModel = BaggingRegressor()
baggingModel.fit(X_train, y_train)


print('Regressor Train Score is : ' ,baggingModel.score(X_train, y_train))
print('Regressor Test Score is : ' , baggingModel.score(X_test, y_test))

y_pred = baggingModel.predict(X_test)


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
print('mean',mean_absolute_error(y_test, y_pred))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
print('mean squre', mean_squared_error(y_test, y_pred))
from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred)
print('median',median_absolute_error(y_test, y_pred))



#print("Accuracy:",accuracy_score(y_test, y_pred))
