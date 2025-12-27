#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15 December 2025

@author: natashatretter
"""  

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # ADDED
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def split_dataset(X, y):

    return train_test_split(X, y, test_size=0.3, random_state=100)


def train_entropy (X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", 
                                         random_state=100, 
                                         class_weight='balanced', 
                                         min_samples_leaf=5, 
                                         max_depth=10
                                         )

    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# ===== ADDED =====
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=100)
    rf.fit(X_train, y_train)
    return rf
# -----------------


def prediction(x_test, clf_entropy, threshold = .45):
    y_prob = clf_entropy.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred


def calc_accuracy(y_test, y_pred):
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy: \n", accuracy_score(y_test, y_pred) * 100)
    print("report: \n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    # replace with where your location is
    df = pd.read_csv('/Users/natashatretter/Documents/Final/Modified_Airlines.csv') # replace with where your location is 
    ################## Sherri's  Code ####################
    df = df.rename(columns={
        'Time': 'departure_minutes',
        'Length': 'duration_minutes',
        'Delay': 'departure_delay',
    })

    df['DayName'] = df['DayOfWeek'].map(
        {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'})

    df['flight_length'] = pd.cut(df['duration_minutes'],
                                 bins=[0, 60, 180, 360, 700],
                                 labels=['Short', 'Medium', 'Long', 'Very Long'])

    df['delay_status'] = df['departure_delay'].map(
        {0: 'On-Time', 1: 'Delayed'})
    df['DepartureFormatted'] = df['departure_minutes'].apply(
        lambda x: f"{x // 60:02d}:{x % 60:02d}")

    df['DurationFormatted'] = df['duration_minutes'].apply(
        lambda x: f"{int(x//60)}h {int(x % 60)}m" if x >= 60 else f"{int(x)}m"
    )

    df['departure_tp'] = pd.cut(df['departure_minutes'],
                                bins=[0, 360, 720, 1080, 1440],
                                labels=['Early Morning', 'Morning', 'Afternoon', 'Evening'])

    df['delay_status'] = df['departure_delay'].map(
        {0: 'On-Time', 1: 'Delayed'})

    print(df)

    df['flight_length'].value_counts()
    ##################################################################
    df_copy = df.copy()
    df_copy['flight_length'] = df_copy['flight_length'].cat.codes 
    df_copy['delay_status'] = df_copy['delay_status'].map({'On-Time': 0, 'Delayed': 1})
    df_copy['departure_tp'] = df_copy['departure_tp'].cat.codes    
    features = ['DayOfWeek', 'departure_minutes', 'duration_minutes', 'flight_length', 'departure_tp']
    target = 'delay_status'
    
    
    X = df_copy[features]
    X_df = df_copy[features]
    y = df_copy[target]

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    clf = train_entropy(X_train, X_test, y_train)
    
    y_pred = prediction(X_test, clf)

    print("=== Decision Tree Results ===")  # ADDED

    calc_accuracy(y_test, y_pred)

    # ===== ADDED =====
    clf_rf = train_random_forest(X_train, y_train)
    y_pred_rf = prediction(X_test, clf_rf)
    print("\n=== Random Forest Results ===")
    calc_accuracy(y_test, y_pred_rf)

    feature_names = X_df.columns

    feature_importances = pd.Series(
        clf_rf.feature_importances_, index=feature_names)
    print("\n=== Random Forest Feature Importances ===")
    print(feature_importances.sort_values(ascending=False))
    # -----------------


