#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 22:10:35 2025

@author: mayaq
"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def split_dataset(balance_data):

    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test

def train_entropy (X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion ="entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(x_test, clf_entropy):
    y_pred = clf_entropy.predict(x_test)
    return y_pred

def calc_accuracy(y_test, y_pred):
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy: \n", accuracy_score(y_test, y_pred) * 100)
    print("report: \n", classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    df = pd.read_csv('/Users/mayaq/Downloads/Airlines.csv') # replace with where your location is 
    ################## Sherri's  Code ####################
    df = df.rename(columns={
        'Time': 'departure_minutes',
        'Length': 'duration_minutes',
        'Delay': 'departure_delay',
        })

    df['DayName'] = df['DayOfWeek'].map({1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'})

    df['flight_length'] = pd.cut(df['duration_minutes'],
                             bins=[0, 60, 180, 360, 700],
                             labels=['Short', 'Medium', 'Long', 'Very Long'])

    df['delay_status'] = df['departure_delay'].map({0:'On-Time', 1:'Delayed'})
    df['DepartureFormatted'] = df['departure_minutes'].apply(lambda x: f"{x // 60:02d}:{x % 60:02d}")


    df['DurationFormatted'] = df['duration_minutes'].apply(
        lambda x: f"{int(x//60)}h {int(x%60)}m" if x >= 60 else f"{int(x)}m"
        )

    df['departure_tp'] = pd.cut(df['departure_minutes'],
                            bins=[0, 360, 720, 1080, 1440],
                            labels=['Early Morning', 'Morning', 'Afternoon', 'Evening'])


    df['delay_status'] = df['departure_delay'].map({0:'On-Time', 1: 'Delayed'})

    print(df)

    df['flight_length'].value_counts()
    ##################################################################
    df_copy = df.copy()
    df_copy['flight_length'] = df_copy['flight_length'].cat.codes # makes short medium Long very long into numerical
    df_copy['delay_status'] = df_copy['delay_status'].map({'On-Time': 0, 'Delayed': 1}) # makes on-time and delayed into numerical
    
    features = ['DayOfWeek', 'departure_minutes', 'duration_minutes', 'flight_length']
    target = 'delay_status'
    
    X = df_copy[features]
    Y = df_copy[target]
    
    X, Y, X_train, X_test, y_train, y_test = split_dataset(pd.concat([Y, X], axis=1))
    
    clf = train_entropy(X_train, X_test, y_train)
    
    y_pred = prediction(X_test, clf)
    
    calc_accuracy(y_test, y_pred)