#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
#import pyspark.pandas as ps
#from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split                                            

__author__ = 'Jacob Hajjar'
__email__ = 'hajjarj@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar'


def perform_regression():
    '''load the data and perform linear regression to predict the price and return the model'''
    df = pd.read_csv("used_cars_labeled.csv")
    cur_make = "Toyota"
    toyota_df = df[df['make'] == cur_make]
    #perform any preprocessing to fix t/f values and clear blank entries
    y_data = toyota_df['price']
    #x_data =  toyota_df.drop(['make', 'price'], axis=1)
    x_data =  toyota_df([['year','model','trim','drivetrain','fuel_type','transmission','mileage','in_accident','1_owner','personal_used']])
    print(np.unique(toyota_df['model'].to_numpy().astype("str"))) 
    #ps_y_data = 
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    #convert to pandas on spark dataframe


def main():
    '''the main function'''
    perform_regression()
    

if __name__ == '__main__':
    main()