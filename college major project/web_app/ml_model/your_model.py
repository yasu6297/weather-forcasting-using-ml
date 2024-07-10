import numpy as np 
from flask import Flask, render_template
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import plotly.express as px 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from neuralprophet import NeuralProphet
from sklearn.preprocessing import LabelEncoder
import pickle

df=pd.read_csv('web_app\ml_model\seattle-weather.csv')


df

# Converting the Dtype on date from object to datetime
df['date'] = pd.to_datetime(df['date'])


# The column weather contains the data value in the string form and we need to predict the weather data 
# so we convert it to an int as label.
df['weather']=LabelEncoder().fit_transform(df['weather'])

# Visualizing the temperature, observing if there's abnormal data
plt.figure(figsize = (15, 5))
fig = plt.plot(df['date'], df[['temp_min', 'temp_max']])
plt.grid();

plt.figure(figsize = (15, 5))
fig2 = plt.plot(df['date'], df[['precipitation', 'wind']])
plt.grid();

df['month'] = df['date'].dt.month_name(locale='English')

fig = px.box(df, df.month, ['temp_min', 'temp_max'])
fig.update_layout(title='Warmest and Coldest Monthly Tempratue.')
fig.show()

df[["precipitation","temp_max","temp_min","wind"]].corr()

plt.figure(figsize=(12,7))
sns.heatmap(df[["precipitation","temp_max","temp_min","wind"]].corr(),annot=True,cmap='coolwarm');

features=["precipitation", "temp_max", "temp_min", "wind"]
X=df[features]
y=df.weather
X_train, X_test, y_train,y_test = train_test_split(X, y,random_state = 0)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
print("XGB Accuracy:{:.2f}%".format(xgb.score(X_test,y_test)*100))

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print("KNN Accuracy:{:.2f}%".format(knn.score(X_test,y_test)*100))

ab = AdaBoostClassifier()
ab.fit(X_train, y_train)
print("AB Accuracy:{:.2f}%".format(ab.score(X_test,y_test)*100))

ab.get_params().keys()

#predicting values from GridSearchCV
y_pred = knn.predict(X_test)

# show classification report on test data
print(classification_report(y_test.values, y_pred, zero_division=1))

filename = 'knn_model.pkl'
pickle.dump(y_pred, open(filename, 'wb'))


