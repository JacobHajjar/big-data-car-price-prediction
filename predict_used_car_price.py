#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
#from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.sql.functions import col

import matplotlib.pyplot as plt 
import seaborn as sns                   

__author__ = 'Jacob Hajjar'
__email__ = 'hajjarj@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar'


def load_process_data():
    '''load the data, process it for use in ml, return x and y dataframe'''
    df = pd.read_csv("used_cars_labeled.csv")
    cur_make = "Honda"
    make_df = df[df['make'] == cur_make]
    make_df  = make_df[['year','model','mileage','in_accident','1_owner','personal_used', 'price']]
    std = StandardScaler()
    make_df['mileage'] = std.fit_transform(make_df['mileage'].to_numpy().reshape(-1, 1))
    make_df['year'] = std.fit_transform(make_df['year'].to_numpy().reshape(-1, 1))
    #convert t/f to bool
    make_df['in_accident'] = make_df['in_accident'].astype('int')
    make_df['1_owner'] = make_df['1_owner'].astype('int')
    make_df['personal_used'] = make_df['1_owner'].astype('int')
    #drop models under 20 entries, then one hot encode
    model_list = make_df['model'].value_counts()
    for model, count in model_list.items():
        if count < 20:
            make_df = make_df[make_df['model'] != model]
    #one hot encode all of the models
    encoded_make_df = pd.get_dummies(make_df, columns=['model'])

    return encoded_make_df

def load_data_spark(spark):
    '''load and process the data using spark instead'''
    data_path = "used_cars_labeled.csv"
    df = spark.read.option("header",True).csv(data_path)
    #set the make of the car data we want to load
    cur_make = "Honda"
    make_df = df[df['make'] == cur_make]
    make_df  = make_df[['year','model','mileage','in_accident','1_owner','personal_used', 'price']]
    #turn y value into int
    make_df = make_df.withColumn('price',col('price').cast("int"))
    #standardize the mileage
    make_df = make_df.withColumn('mileage',col('mileage').cast("int"))
    vectorAssembler = VectorAssembler(inputCols = ['mileage'], outputCol = 'vector_mileage')
    make_df = vectorAssembler.transform(make_df)
    std = StandardScaler(inputCol='vector_mileage', outputCol='scale_mileage')
    make_df = std.fit(make_df).transform(make_df)
    make_df.show()
    #standardize the year
    make_df = make_df.withColumn('year',col('year').cast("int"))
    vectorAssembler = VectorAssembler(inputCols = ['year'], outputCol = 'vector_year')
    make_df = vectorAssembler.transform(make_df)
    std = StandardScaler(inputCol='vector_year', outputCol='scale_year')
    make_df = std.fit(make_df).transform(make_df)
    #old way of doing it in pandas
    #make_df['mileage'] = std.fit_transform(make_df['mileage'].to_numpy().reshape(-1, 1))
    #make_df['year'] = std.fit_transform(make_df['year'].to_numpy().reshape(-1, 1))
    #convert t/f to bool
    '''make_df['in_accident'] = make_df['in_accident'].astype('int')
    make_df['1_owner'] = make_df['1_owner'].astype('int')
    make_df['personal_used'] = make_df['1_owner'].astype('int')'''

    make_df = make_df.withColumn('in_accident',col('in_accident').cast("boolean"))
    make_df = make_df.withColumn('in_accident',col('in_accident').cast("integer"))

    make_df = make_df.withColumn('1_owner',col('1_owner').cast("boolean"))
    make_df = make_df.withColumn('1_owner',col('1_owner').cast("integer"))

    make_df = make_df.withColumn('personal_used',col('personal_used').cast("boolean"))
    make_df = make_df.withColumn('personal_used',col('personal_used').cast("integer"))

    #make_df = make_df.select([col(c).cast("integer") for c ["test1", "test2"]])
    make_df.show()
    #drop models under 20 entries, then one hot encode
    '''model_list = make_df['model'].value_counts()
    for model, count in model_list.items():
        if count < 20:
            make_df = make_df[make_df['model'] != model]'''
    #one hot encode all of the models
    #encoded_make_df = pd.get_dummies(make_df, columns=['model'])
    #vectorAssembler = VectorAssembler(inputCols = ['model'], outputCol = 'vector_model')
    #make_df = vectorAssembler.transform(make_df)

    stringIndexer = StringIndexer(inputCol="model", outputCol="model_index")
    make_df = stringIndexer.fit(make_df).transform(make_df)
    print(stringIndexer.params)
    make_df.show()
    encoder = OneHotEncoder(inputCol="model_index", outputCol='encoded_model')
    encoded_make_df = encoder.fit(make_df).transform(make_df)
    encoded_make_df.show()

    return encoded_make_df


def perform_regression(sqlContext, dataset):
    '''load the data and perform linear regression to predict the price and return the model'''
    #get the data's columns and delete the prediction column which is price
    #feature_columns = np.array(dataset.columns)
    #feature_columns = np.delete(feature_columns, np.where(feature_columns == 'price')[0])


    #make_spark_df = sqlContext.createDataFrame(dataset)
    make_spark_df = dataset
    feature_columns = ['in_accident', '1_owner', 'personal_used', 'scale_mileage', 'scale_year', 'encoded_model']
    #feature_columns = ['in_accident', '1_owner', 'personal_used', 'scale_mileage', 'scale_year']
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

    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    lr_predictions = lr_model.transform(test_df)

    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="price",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(test_df)
    print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
    #plot results to analyze most significant features
    print(len(lr_model.coefficients))
    print(len(feature_columns))
    coefs = pd.DataFrame(lr_model.coefficients, columns=['Coefficients'], index=feature_columns)
    coefs.plot(kind='barh', figsize=(9, 7))
    plt.title('Ridge model')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.show()


def main():
    '''the main function'''
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    #conf = SparkConf().setMaster("local").setAppName("BigDataAnalysis")
    #sc = SparkContext(conf = conf)
    #sqlContext = SQLContext(sc)

    #create the spark session
    spark = SparkSession.builder.master("local").appName("BigDataAnalysis").getOrCreate()
    #enable pyarrow optimizations with spark when converting to and from dataframes
    #spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    #make_df = load_process_data()
    make_df = load_data_spark(spark)
    perform_regression(spark, make_df)

if __name__ == '__main__':
    main()