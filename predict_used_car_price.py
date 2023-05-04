#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import VectorAssembler                     

__author__ = 'Jacob Hajjar'
__email__ = 'hajjarj@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar'


def load_process_data():
    '''load the data, process it for use in ml, return x and y dataframe'''
    df = pd.read_csv("used_cars_labeled.csv")
    cur_make = "Toyota"
    toyota_df = df[df['make'] == cur_make]
    #perform any preprocessing to fix t/f values and clear blank entries
    #y_data = toyota_df['price']
    #x_data =  toyota_df.drop(['make', 'price'], axis=1)
    #x_data =  toyota_df[['year','model','drivetrain','fuel_type','transmission','mileage','in_accident','1_owner','personal_used']]
    #x_data =  toyota_df[['year','model','mileage','in_accident','1_owner','personal_used']]
    #standardize year and mileage
    toyota_df  = toyota_df[['year','model','mileage','in_accident','1_owner','personal_used', 'price']]
    std = StandardScaler()
    toyota_df['mileage'] = std.fit_transform(toyota_df['mileage'].to_numpy().reshape(-1, 1))
    toyota_df['year'] = std.fit_transform(toyota_df['year'].to_numpy().reshape(-1, 1))
    #convert t/f to bool
    toyota_df['in_accident'] = toyota_df['in_accident'].astype('int')
    toyota_df['1_owner'] = toyota_df['1_owner'].astype('int')
    toyota_df['personal_used'] = toyota_df['1_owner'].astype('int')
    #drop models under 20 entries, then one hot encode
    model_list = toyota_df['model'].value_counts()
    for model, count in model_list.items():
        if count < 20:
            toyota_df = toyota_df[toyota_df['model'] != model]

    encoded_make_df = pd.get_dummies(toyota_df, columns=['model'])

    #toyota_df = toyota_df[toyota_df['model'] == 'Corolla']
    #print(np.unique(toyota_df['drivetrain'].to_numpy().astype("str"), return_counts=True)) 
    #convert drivetrain to 4 categories, RWD, FWD, 4WD, AWD, and remove unknown, then one hot encode

    #process transmission to 0 for automatic or 1 for manual, or drop

    #remove fuel_type under 20
    return encoded_make_df

def perform_regression(dataset):
    '''load the data and perform linear regression to predict the price and return the model'''

    #ps_x_data = ps.from_pandas(x_data)
    #ps_y_data = ps.from_pandas(y_data)
    #ps_data = ps.from_pandas(toyota_df)
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    feature_columns = np.array(dataset.columns)
    #print(feature_columns)
    feature_columns = np.delete(feature_columns, np.where(feature_columns == 'price')[0])
    conf = SparkConf().setMaster("local").setAppName("BigDataAnalysis")
    sc = SparkContext(conf = conf)
    sqlContext = SQLContext(sc)
    make_spark_df = sqlContext.createDataFrame(dataset)
    vectorAssembler = VectorAssembler(inputCols = feature_columns, outputCol = 'features')
    make_spark_df = vectorAssembler.transform(make_spark_df)
    make_spark_df = make_spark_df.select(['features', 'price'])

    make_spark_df_split = make_spark_df.randomSplit([0.7, 0.3])
    train_df = make_spark_df_split[0]
    test_df = make_spark_df_split[1]

    lr = LinearRegression(maxIter=50, regParam=0.3, elasticNetParam=0.8, labelCol='price')
    lr_model = lr.fit(train_df)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    #convert to pandas on spark dataframe


def main():
    '''the main function'''
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable 
    toyota_df = load_process_data()
    perform_regression(toyota_df)
    

if __name__ == '__main__':
    main()