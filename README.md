# Fraud_Detection
The Covid-19 pandemic, which we have been struggling with for about 2 years, has also changed our shopping habits. While the habit of going to the store has almost come to an end, online shopping has started to rise rapidly. There are great dangers waiting for the consumer who buys everything from the internet, from furniture to clothes, from food to cosmetics. Credit card fraud!

Credit card fraud is one of the most common online scams. Credit card numbers, PINs, and security codes can be easily stolen and used to make fraudulent transactions. This can cause huge financial losses for traders and consumers. However, credit card companies are ultimately obligated to reimburse their customers for any losses. Therefore, it is extremely important that credit card companies and other financial institutions can detect fraud before it occurs.
Machine learning is considered the most reliable method for detecting fraudulent transactions. By training a machine learning model with a real dataset, fraudulent transactions can be quickly detected in real-time.

We will train the ML model with a dataset of a Kaggle. The dataset belongs to a Kaggle competition.
- The competition was held 2 years ago and 6351 teams participated.
- First prize was awarded $10.000, a second prize with $7.000, and the third prize with $3.000.
- The competition started on July 15, 2019, and ended on October 3, 2019, lasting 50 days.
The LightGBM and XGBoost models, which won many competitions in Kaggle, proved how successful they were. In this study, I will train the model with LightGBM and XGBoost and compare their performances with each other.

## About Dataset
Data comes from Vesta’s (a transaction guarantee platform for digital purchases) real-world e-commerce transactions and includes a wide variety of attributes, from device type to product specifications.

## Data Processing and Understanding
Firstly, we load the libraries that may be needed. Train and test datasets are given in 2 pieces. We import the data in CSV format using pandas and merge it over the common column.

## Features
isFraud : binary, Target
TransactionID : all unique. It is pure noise right now. Since almost all values of ‘TransactionID’ are unique, it is dropped.
TransactionDT : time series. Time from reference time point. VERY valuable column
TransactionAmt : continuous. It has many unique values and has to be combined with other columns. The best score boost should come from TransactionDT->TransactionAmt combination
P_emaildomain : categoric, 56 uniques. It's possible to make subgroup feature from it or general group
R_emaildomain : categoric, 59 uniques. It's possible to make subgroup feature from it or general group
DeviceType : categoric, 2 uniques
DeviceInfo : categoric, 700 uniques
ProductCD : categoric, 5 uniques
100% categorical feature options to use: Frequency encoding/Target encoding/Combinations with other columns/Model categorical feature
card1-6 : categoric, numeric. Categorical features with information about Client
addr1-2 : addr1 - subzone / add2 - Country
dist1-2 : numeric. dist1 - local distance from merchant / dist2 - Country distance
D1-15 : numeric. time delta, such as days between the previous transactions, etc. The minimal value will be the same for each month and day but maximum and mean values will grow over time.
C1-14 : numeric
M1-9 : categoric
V1-339 : numeric, categoric
id_01_38 : numeric, categoric

So we have two medium-sized datasets with a lot of columns. The most important thing you’ll notice about this data is that the dataset is unbalanced with respect to one feature. We can see that the majority of transactions in our datasets are normal and only a few percent of transactions are fraudulent. Let’s check the transaction distribution.

3.5% of transactions are fraud. Notice how imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this data frame as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will “assume” that most transactions are not fraudulent. But we don’t want our model to assume, we want our model to detect patterns that give signs of fraud! Imbalance means that the number of data points available for different classes is different.

Working with large data while training ML Models requires large RAM memory. To overcome this limitation, we used a function to reduce the memory footprint of the data. The general approach is to convert dtype of each feature (‘int16’, ‘int32’, ‘int64’, ‘float16’, ‘float32’, ‘float64’) to the lowest possible dtype.
Objects created in RunTime are cleared from memory when they are not needed by the application or when the object created in the program is finished. The “Garbage Collector” (gc.collect())is used for this process.

Columns containing only 1 unique value and containing more than 90 percent missing values and when missing values are dropped, columns with more than 90 percent of the remaining data are dropped from the dataset.

While train datasets are having observations for 182 days, test datasets are having observations for 183 days. There is a gap of 30 days between the two datasets. We can generate some useful time-features from TransactionDT, the time from the reference time point. It is a very valuable column.

## Feature Engineering
According to Wikipedia, feature engineering refers to the process of using domain knowledge to extract features from raw data via data mining techniques. These features can then be used to improve the performance of machine learning algorithms. Feature engineering does not necessarily have to be fancy though. One simple, yet prevalent, use case of feature engineering is in time-series data. The importance of feature engineering in this realm is due to the fact that (raw) time-series data usually only contains one single column to represent the time attribute, namely date-time (or timestamp). Regarding this date-time data, feature engineering can be seen as extracting useful information from such data as standalone (distinct) features. For example, from a date-time data of “2020–07–01 10:21:05”, we might want to extract the following features from it:
Month: 7
Day of month: 1
Day name: Wednesday (2020–07–01 was Wednesday)
Hour: 10
The minimum value in the D columns has to equal the minimum value for each month and the first day of the month. In order to meet this condition, the START_DATE has been determined as ‘2017–11–30’.

We generated 8 different time features from one feature that includes the reference time.

In the morning hours, the fraud rate was high. Afternoon, It decreased.
In December, the Fraud rate decreased. December is the common month for both train and test datasets. This is an important point.
The fraud rate decreases towards the last days of the month while decreasing during the holidays.
The fraud rate tends to increase from the beginning of the week, tends to decrease on Friday and Sunday.

The fraud rate tends to increase from the beginning of the week, tends to decrease on Friday and Sunday.Similar to the weekly distribution, there is a similar distribution within the month. While the fraud rate is low at the beginning of the month, this rate tends to increase towards the end of the month.
While the fraud rate is low on national public holidays in the USA, the fraud rate is proportionally higher on other days.

After this stage, let’s list the actions we have done and the insight we have extracted in the project.
1. There is no dramatic difference between the TransactionAmt averages of the train and test datasets. The average of the fraud transactions(149.24) is bigger than the average of the non-fraud transactions(134.26).

2. Test dataset has some new software version and device that train dataset doesn’t have. And also train dataset has a software version and device that the test dataset doesn’t have. We replace them with Nan value.

3. id_28and id_29are having categorical data. having a high correlation. (Treshold= 0.9). So we dropped id_29.Click on the link for detailed information about the correlation of categorical data.

4. We generated 2 new columns with mail server and domain.
5. addr1 and addr2 are related to addressing. add2 — Country / addr1 — subzone. By combining them, we create a new column.

V columns seem redundant and interrelated. Also, many subsets have similar NAN structures. We determined the columns with correlation (r > 0.75) in the V columns and dropped one of them, leaving only the independent columns. After dropping the correlated V columns, we have only 62 V columns.

## PCA for V Columns
With so many features, the performance of your algorithm will drastically degrade. PCA (Principal Component Analysis) is a very common way to speed up your Machine Learning algorithm by getting rid of correlated variables which don’t contribute to any decision making. The training time of the algorithms reduces significantly with less number of features. In addition, Overfitting mainly occurs when there are too many variables in the dataset. So, PCA helps in overcoming the overfitting issue by reducing the number of features.

So, if the input dimensions are too high, then using PCA to speed up the algorithm is a reasonable choice. Now we will apply PCA to the V columns of 62 features. But first, we have to normalize the V columns. We are looking for a requirement that it represents more than 90% of the data. We achieved 92% representation of V Columns with 3 PCA variables. Very good!

## Frequency Encoding
There are many ways of dealing with categorical columns. One hot encoding is a good way, but if feature cardinality is high, then we will have too many new columns and could hit memory limits. One hot encoding was not used to get rid of the curse of multidimensionality. Although label encoding is a good solution for ordinal data, it is not a suitable solution for nominal data. It can be said that frequency encoding is the best solution for this dataset.

## Handling Outliers
Dropping or manipulating outliers in Fraud detection projects will not generally allow the model to predict better. Because Fraud transactions themselves can be considered as outliers. That is, we want the model to detect outlier observations. For these reasons, the approach of leaving the outliers as they are has been adopted in this project.

## Handling Missing Values
LightGBM and XGBoost Libraries can handle missing values
LightGBM: will ignore missing values during a split, then allocate them to whichever side reduces the loss the most
XGBoost: the instance is classified into a default direction (the optimal default directions are learned from the data somehow)
It is NOT a general property of gradient descent algorithms or tree algorithms. Only specific implementations of these algorithms have this property. In this project, we left the missing values to the evaluation of the model and did not make any assignments.

## Modeling with LightGBM
Gradient Boosting Decision Trees (GBDT) algorithms have been proven to be among the best algorithms in machine learning. XGBoost, the most popular GBDT algorithm, has won many competitions on websites like Kaggle. However, XGBoost is not the only GBDT algorithm with state-of-the-art performance. There are other GBDT algorithms that have more advantages than XGBoost and are even more potent like LightGBM and CatBoost.
While standardizing data for linear models is required, for tree-based models, it is mostly useless. Tree-based models make splits between values of variables, so exact values matter little. The difference could be if we limit the number of bins to analyze. The same for correlation. It is important to analyze it for linear models, but for tree-based models feature interactions are important. While a feature could have little correlation with the target, it could have a serious effect, when combined with some other variable. In the project, we will first train our model with LightGBM and then with XGBoost Classifier. So we didn’t normalize the data.

While splitting train and test data, it is important to keep the 3.5% rate of the target column the same in train and test. It was provided with a stratify=y parameter.

## Feature Importance for LightGBM Classifier
There was a balanced distribution of feature importance, it seems that there is not. The most powerful feature of boosting algorithms is to work on different sub-observations and sub-independent features and reduce bias.

## Evaluation Metrics
As seen on the left, LightGBM has done much better than XGBoost. It is also more successful than XGBoost in terms of training time and lower memory usage.
Evaluation of the Metrics
Accuracy is a metric that is widely used to measure the success of a model for balanced datasets but our dataset is extremely unbalanced. So, We don’t take into account the accuracy score.
Precision shows how many of the values we estimated as positive are actually positive.
Recall, is a metric that shows how much of the operations we need to estimate as Positive, we estimate as Positive. It is an extremely important score.
F1 Score value shows us the harmonic mean of Precision and Recall values. It is an important score in any kind of balanced or unbalanced dataset.
AUC is one of the most important evaluation scores for checking the performance of any classification model. It is one of the most widely used metrics to evaluate the performance of machine learning algorithms, especially in situations with unbalanced datasets. And it explains how well the model predicts.
ROC is a probability curve for different classes. A typical ROC curve has False Positive Rate (FPR) on the X-axis and True Positive Rate (TPR) on the Y-axis.
Here are a few of the benefits of using a ROC curve:
1.Curves of different models can be directly compared in general or for different thresholds.
2.The area under the curve (AUC) can be regarded as a summary of model skill, in other words, model performance.
3.Usually, successful models are represented by curves that curve to the upper left of the plot.
In the related Kaggle competition, the evaluation score is the AUC score. We got a 0.9535 AUC score with LightGBM and 0.8820 AUC score with XGBoost.