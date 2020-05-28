"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande

Description: Obtain the mortaltiy rate for each Chronic obstructive pulmonary disease (COPD) per year and county.

Concepts Used (Overall): Spark, Hadoop, Multivariate Linear Regression, Feature Extraction
                         Hypothesis Testing, Time Series Prediction

Concepts in this file: Data processing - Data Cleaning, Spark, Hadoop

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB

Sample Output: (In overallresult - line number 51)

('49043', '511', '1', '1984', '1.13271410113499')
('49043', '511', '1', '1985', '1.07641849421599')
('49043', '511', '1', '1986', '1.04215524929848')
('49043', '511', '1', '1987', '1.03671409930344')
('49043', '511', '1', '1988', '0.969186492560371')

It follows ->  CountyId, Disease, Gender, Year, Mortality Rate
"""

from pyspark.sql import SparkSession
import sys
import csv
from itertools import islice


# Pollutant files to be processed
fileMortality = sys.argv[1]

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

rdd = sc.textFile(fileMortality)\
    .mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

# Modify the FIPS code
def append(x):
  x = str(int(float(x)))
  if len(x) == 4:
    x = '0' + x
  return x

rdd.filter(lambda x: x[4] != "United States")\
  .map(lambda x: (append(x[5]), x[6], x[8], x[12], x[14]))\
    .filter(lambda x: len(x[0]) > 2 )\
      .saveAsTextFile('processed_mortality')