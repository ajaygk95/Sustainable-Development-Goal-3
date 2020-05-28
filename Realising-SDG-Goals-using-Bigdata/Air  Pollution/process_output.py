"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: Merge the obtained results (for different pollutants )into a single file

Concepts Used (Overall): Spark, Hadoop, Multivariate Linear Regression, Feature Extraction
                         Hypothesis Testing, Time Series Prediction

Concepts in this file: Data processing - Final result prep, Spark, Hadoop

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB

Sample Output: (In overallresult - line number 55)

(36045, '44201', 1986, 44.7, 122.74276886310433, 0.12739449903029898)
(36045, '44201', 1987, 60.3155737704918, 124.82483242999346, 0.12739449903029898)
(36045, '44201', 1988, 60.69834710743802, 125.99056926936919, 0.12739449903029898)
(36045, '44201', 1989, 53.52173913043478, 121.93412053306619, 0.12739449903029898)

It follows -> county ID, Pollutant ID, Year, AQI, Mortality Rate, Correlation
"""

from pyspark.sql import SparkSession
import glob
import pandas as pd
import sys

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

# Result of Linear Regression merged into a single file.
regressionResultFiles = sys.argv[1]

overallRDD = sc.textFile(regressionResultFiles)

def parseIntoCSVrows(x):
  output = []
  county = int(x[0])
  pollutants = x[1]
  for p in pollutants:
    pollutant = p[0]
    years = p[1]
    aqi = p[2]
    dr = p[3]
    beta = p[4]
    for y in range(len(years)):
      output.append((county, pollutant, years[y], aqi[y], dr[y], beta))
  return output

overallRDD.map(lambda x: eval(x)).groupByKey().mapValues(list)\
.flatMap(parseIntoCSVrows).saveAsTextFile('/overallResult')

paths = glob.glob('/overallResult')

x = []
for path in paths:
  with open(path) as f:
      for line in f:
        x.append(eval(line))
db = pd.DataFrame(x)

db.to_csv('pollutant.csv')
