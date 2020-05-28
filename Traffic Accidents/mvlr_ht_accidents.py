"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: Performed Multivariate Linear regression to fit extracted features with severity of the accidents.
             Performed Hypothesis Testing on the results.

Concepts Used (Overall): Spark, Hadoop, Multivariate Linear Regression, Feature Extraction
                         Hypothesis Testing, Time Series Prediction

Concepts in this file: Multivariate Linear Regression, Hypothesis Testing, Spark, Hadoop

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB

Sample Output:

('1-0-23',
 array([ 0.1953718 ,  0.13067577,  0.25432685,  0.17172011,  0.82247048,
        -0.32328639,  0.1149682 , -0.04738236,  0.50077324,  0.27765478]),
 [8.668711281264457e-06, 0.0008762447049606971, 1.305886449856624e-07, 4.796106911313891e-05, 2.162769204555401e-19,
  1.3295478302889107e-09, 0.0025218988088426403, 0.11005175924260707, 6.865257001169748e-14, 2.64250391125718e-08])

It follows: county, list of betas, list of p-values
"""

import csv
import datetime
import numpy as np
import sys
from scipy import stats
from pyspark import SparkContext
from itertools import islice

sc = SparkContext('local[*]', 'pyspark tutorial')

groupedAccidentByCounty = sys.argv[0]
importantColFile = sys.argv[1]

def mult_linear_reg(line, indicesToKeep, resultCol):
   
    county = line[0]
    column_wise_data = np.transpose(np.asarray(line[1], dtype = object))

    X_trans = []
    Y = []
    # Keep only the features that are marked as significant for that county,
    for index, tup in enumerate(column_wise_data):
      if index in indicesToKeep:
        X_trans.append(tup)
    
    Y = column_wise_data[resultCol]
    X = np.transpose(X_trans)

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_normT = X_norm.T
    Y_norm = (Y - np.mean(Y)) / np.std(Y)

    inv = np.linalg.pinv(np.dot(X_normT, X_norm))
    betas = np.dot(np.dot(inv, X_normT), Y_norm)

    p_val = calc_p_value(county, X_norm, Y_norm, betas)
    return county, betas, p_val


# Hypothesis Testing
def calc_p_value(county, X_norm, Y_norm, betas):
    m = X_norm.shape[0]

    df = m - (1 + X_norm.shape[1])
    rss = np.sum(np.square(Y_norm - np.dot(X_norm, betas)))
    s2 = rss / df
    se_b1 = np.sqrt(s2 / np.sum(X_norm ** 2))
    
    pval = []
    # Loop through all the features to calculate multiple p values
    for i in range(X_norm.shape[1]):
      b_1 = betas[i]
      t_1 = b_1 / se_b1
      pval.append(stats.t.sf(np.abs(t_1), df))
    
    return pval

# From the fars_spark_get_features_countwise.py
accidentRDD = sc.textFile(groupedAccidentByCounty)

# From the fars_spark_get_features_countwise.py
importantCols = sc.textFile(importantColFile).collect()

countyImportantFeatureMapping = {}
for i in importantCols:
  countyImportantFeatureMapping[i[0]] = i[1]

# Broadcast the important features
br_countyImportantFeatureMapping = sc.broadcast(countyImportantFeatureMapping)

# Result column is 45 - Severity of the accident
accidentRDD.map(lambda x:  mult_linear_reg(x, br_countyImportantFeatureMapping[x[0]], 45)).saveAsTextFile('finalResult')

