#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler                                         

__author__ = 'Jacob Hajjar'
__email__ = 'hajjarj@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar'


def load_process_data():
    '''load the data, process it for use in ml, return x and y dataframe'''
    df = pd.read_csv("used_cars_labeled.csv")
    cur_make = "Toyota"
    toyota_df = df[df['make'] == cur_make]
    #perform any preprocessing to fix t/f values and clear blank entries
    y_data = toyota_df['price']
    #x_data =  toyota_df.drop(['make', 'price'], axis=1)
    #x_data =  toyota_df[['year','model','drivetrain','fuel_type','transmission','mileage','in_accident','1_owner','personal_used']]
    x_data =  toyota_df[['year','model','mileage','in_accident','1_owner','personal_used']]
    #standardize year and mileage

    std = StandardScaler()
    x_data['mileage'] = std.fit_transform(x_data['mileage'].to_numpy().reshape(-1, 1))
    x_data['year'] = std.fit_transform(x_data['year'].to_numpy().reshape(-1, 1))
    print(x_data)
    #drop models under 20 entries, then one hot encode
    #toyota_df = toyota_df[toyota_df['model'] == 'Corolla']
    #print(np.unique(toyota_df['drivetrain'].to_numpy().astype("str"), return_counts=True)) 
    #convert drivetrain to 4 categories, RWD, FWD, 4WD, AWD, and remove unknown, then one hot encode

    #process transmission to 0 for automatic or 1 for manual, or drop

    #remove fuel_type under 20
    return x_data, y_data

def perform_regression():
    '''load the data and perform linear regression to predict the price and return the model'''
    x_data, y_data = load_process_data()
    ps_x_data = ps.from_pandas(x_data)
    ps_y_data = ps.from_pandas(y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    lr = LinearRegression(maxIter=50, regParam=0.3, elasticNetParam=0.8)
    lr.setFeaturesCol()
    #convert to pandas on spark dataframe


def main():
    '''the main function'''
    perform_regression()
    

if __name__ == '__main__':
    main()