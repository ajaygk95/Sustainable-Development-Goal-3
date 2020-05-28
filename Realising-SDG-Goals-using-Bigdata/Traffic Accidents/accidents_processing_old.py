"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: We cleaned and processed the data but did not contain any data wrt mortality (fatality).
             This dataset only had accident details.
"""

import os
from pyspark import SparkContext
import csv
import datetime
import numpy as np
import sys
from itertools import islice

fileFile = sys.argv[1]
usAccidentsFile = sys.argv[2]

sc = SparkContext('local[*]', 'pyspark tutorial')

fipsCodes = sc.textFile(fileFile).mapPartitionsWithIndex(
lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

fipsCode = fipsCodes.filter(lambda x: int(x[0])> 1000).map(lambda x: (x[1], int(x[0]), x[2])).collect()

fipsMapping = {}
for i in fipsCode:
  fipsMapping[i[0]] = [i[1], i[2]]

fipsCode = sc.broadcast(fipsMapping)

def extractDayOfTheYear(d):
  currDate = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
  return (currDate - datetime.datetime(currDate.year,1,1)).days

def accidentInterval(s, e):
  sd = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()
  ed = datetime.datetime.strptime(e, '%Y-%m-%d %H:%M:%S').timestamp()
  return ((ed-sd)/3600)

def modifySide(x):
  if x == 'R':
    return 1
  else:
    return 0

def modifyDataSet(x):
  result = []

  year = datetime.datetime.strptime(x[4], '%Y-%m-%d %H:%M:%S').year
  month = datetime.datetime.strptime(x[4], '%Y-%m-%d %H:%M:%S').month
  dateOfYear = extractDayOfTheYear(x[4])
  interval = round(accidentInterval(x[4], x[5])*100)/100

  result.append(year)
  result.append(month)
  result.append(dateOfYear)
  result.append(interval)
  
  result.append(int(x[3]))
  result.append(float(x[6]))
  result.append(float(x[7]))
  result.append(modifySide(x[14]))
  
  removeCols = [1,2,3,5,6,7,8,9,10,11,14,16,17,18,19,20,21,22,28]
  
  for i in range(0, len(x)):
    if not i in removeCols:
      if i in range(32,45):
        if x[i] == 'False':
          result.append(0)
        else:
          result.append(1)
      else:
        result.append(x[i])
  
  # result.append(x[16])
  return (fipsCode.value[x[16] + ' County'][0], (result))

rdd = sc.textFile(usAccidentsFile).mapPartitionsWithIndex(
lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

rdd.map(lambda x: modifyDataSet(x)).saveAsTextFile('processedResult')
