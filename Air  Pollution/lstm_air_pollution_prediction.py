"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: Prediction of AQI (Air Quality Index) for air pollutants based on data from 1980 to 2019

Concepts Used (Overall): Spark, Hadoop, Multivariate Linear Regression, Feature Extraction
                         Hypothesis Testing, Time Series Prediction

Concepts in this file: Time Series Prediction using LSTM (Tensorflow backend), Spaek, Hadoop

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

import os
import sys
from pyspark.sql import SparkSession
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

pollutantConc = sys.argv[0]
rdd = sc.textFile(pollutantConc)

def split_data(x):
  data = x.split(',')
  return data[0], (data[1:],)

def sorted_data(data):
  county = data[0]
  sort_data = sorted(data[1], key=lambda x: x[0])
  X = []
  for s in sort_data:
    X.append(float(s[1]))

  return county, X

group_by_county = rdd.map(split_data).reduceByKey(lambda x,y : x+y).map(sorted_data)

# As an exmaple consider county '06037 - Los Angeles'
one_county = group_by_county.filter(lambda x: x[0] == '06037').take(1)[0]

def forecast(data):
  county = data[0]
  dataset = data[1]
  epochs= 10

  # reframed = series_to_supervised(dataset, 8, 8)
  # values = reframed.values
  X = []
  y = []
  for i in range(len(dataset)-15):
    X.append([dataset[j] for j in range(i, i+15)])
    # y.append([dataset[j] for j in range(i+10,i+15)])

  train = np.array(X[:-3000])
  test = np.array(X[-3000:])

#   n_obs = 8 * 13
  train_X, train_y = train[:, :-5], train[:, -5:]
  test_X, test_y = test[:, :-5], test[:, -5:]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  model = Sequential()
  model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(5))
  model.compile(loss='mse', optimizer='adam')
  # fit network
  history = model.fit(train_X, train_y, epochs=epochs, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

  yhat = model.predict(test_X)

  avg_preds = {} 
  for idx, pred8 in enumerate(yhat):
      for ind, pred in enumerate(pred8):
          val = avg_preds.get(ind + idx, 0)
          avg_preds[ind+idx] = pred


  pred = [v for k, v in avg_preds.items()]

  return county, pred

pred = forecast(one_county)

plt.subplots(figsize=(20,8))
plt.plot(one_county[1][-3000:])
plt.plot(pred[1])
plt.show()