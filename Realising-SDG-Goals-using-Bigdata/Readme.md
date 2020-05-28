#### Introduction 

The main goal of the project is to analyse the existing datasets and find patterns which identify the factors contributing to deaths due to Air pollution and Traffic accidents in the US and propose ideas to achieve SDG-3 Goal.

The project aligns with the SDG 3 (by WHO) which aims to ensure healthy lives and promote well-being for all, at all ages. We have mainly focused on the following aspects of SDG:
- By 2030, halve the number of global deaths and injuries from road traffic accidents.
- By 2030, substantially reduce the number of deaths and illnesses from hazardous chemicals and air, water and soil pollution and contamination.

#### Data Used

- Pollution dataset: EPA - 1.2 GB
- Fatal traffic accidents: FARS -  3.83 GB
- Accident dataset: Kaggle -  1 GB
- Chronic Respiratory Diseases Mortality Data: GHDx - 700 MB
- Mortality dataset: CDC - 4 GB

#### Methods

The below-described methods are performed on Google Cloud Platform. We have used a standard cluster with 1 Master (e2-standard, 2 cores, 32GB) node and 3 Worker nodes (e2-standard, 4 cores, 64GB) which primarily runs HDFS and Spark on Yarn in cluster mode.

1) Air Pollution Analysis Pipeline

![image](https://user-images.githubusercontent.com/17957548/82724908-4b98d480-9ca7-11ea-9122-bd84f6ad46c1.png)

2) Accidents Analysis Pipeline

![image](https://user-images.githubusercontent.com/17957548/82724919-6703df80-9ca7-11ea-9245-3c989506ea29.png)

#### Some Results


Air pollutants vs Mortality rates for San Francisco county: 

More results can be found <a href="https://sdg-angular.uk.r.appspot.com/"> here </a>

![image](https://user-images.githubusercontent.com/17957548/82724953-a5010380-9ca7-11ea-9b0d-66430a67b88e.png)

Forecasting using LSTM:

![image](https://user-images.githubusercontent.com/17957548/82724990-e42f5480-9ca7-11ea-9cae-582771bd152c.png)

Some analysis on Accident dataset:

![image](https://user-images.githubusercontent.com/17957548/82724998-f6a98e00-9ca7-11ea-985b-15b5dd99cea8.png)






