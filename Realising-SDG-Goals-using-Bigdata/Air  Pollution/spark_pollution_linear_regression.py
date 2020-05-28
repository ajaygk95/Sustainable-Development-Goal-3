"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: Perform Single Variable Linear Regression to find the correlation
             between pollutants and mortality rate.

Concepts Used (Overall): Spark, Hadoop, Multivariate Linear Regression, Feature Extraction
                         Hypothesis Testing, Time Series Prediction

Concepts in this file: Linear Regression, Spark, Hadoop

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

import numpy as np
import sys
from pyspark.sql import SparkSession


spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

# Pollutant (wildcard) to be processed
pollutant = sys.argv[1]

# File that has processed death rate 
deathrateFile = sys.argv[2]

pollutantRdd = sc.textFile(pollutant)

def split_data(x):
  data =  x.split(',')
  return data[0], (data[1:],)

def aggregate_on_year(x):
  fips = x[0]
  data = x[1]
  avg_aqi = {}
  output= []
  for yr in range(1980,2020):
    avg_aqi[yr] = []
  for d in data:
    yr = int(d[0].split('-')[0])
    avg_aqi[yr].append(float(d[1]))
  
  for d in avg_aqi.keys():
    output.append((d, np.mean(avg_aqi[d])))
  return fips, output

avg_aqi_county = pollutantRdd.map(split_data).reduceByKey(lambda x,y: x+y).map(aggregate_on_year)

avg_aqi_county.saveAsTextFile('intermediate-result' + pollutant)

#################################################
# Join it with Mortality
#################################################

death_rdd = sc.textFile(deathrateFile)

def parse_data(x):
  x = x.replace("'" , '').replace('(', '').replace(')', '').replace(' ', '').split(',')
  fips = x[0].split('.')[0]
  # cause = int(x[1])
  typ = int(x[2])
  year = int(x[3])
  mean_death = float(x[4])
  return (fips, year) , [typ, mean_death]

total_death_yr_county = death_rdd.map(parse_data)\
    .filter(lambda x: x[1][0] == 3)\
        .map(lambda x : (x[0], x[1][1])).reduceByKey(lambda x,y : x+y)
  
county_yr_death = total_death_yr_county\
    .map(lambda x: (x[0][0], [(x[0][1],x[1])]))\
        .reduceByKey(lambda x,y : x+y)

county_aqi_death = avg_aqi_county.join(county_yr_death)

#################################################
# Linear Regression
#################################################

def linear_reg(x):
  fips = x[0]
  aqi = x[1][0]
  death = x[1][1]

  aqi = dict(aqi)
  death = dict(death)

  yr = []
  X = []
  Y = []

  for k in aqi.keys():
    if not np.isnan(aqi[k]):
      r = death.get(k, None)
      if r:
        yr.append(k)
        X.append(aqi[k])
        Y.append(r)

  X = np.array(X)
  Y = np.array(Y)
  
  if len(X) < 10:
    return fips, (pollutant, yr, X.tolist(), Y.tolist(), None)

  X_norm = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
  X_normT = X_norm.T
  Y_norm = (Y-np.mean(Y)) / np.std(Y)

  inv = 1 / (np.dot(X_normT,X_norm))
  betas = np.dot(np.dot(inv , X_normT) , Y_norm)
  return fips, (pollutant, yr, X.tolist(), Y.tolist(), betas)

county_aqi_death.map(linear_reg).saveAsTextFile('Linear-Regression' + pollutant)