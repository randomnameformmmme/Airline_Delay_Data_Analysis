#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        #used to store training data
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        #predicting labels for test data
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        #Calculate distances to all training points
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        #Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        #Get labels of k nearest neighbors
        k_labels = [self.y_train[i] for i in k_indices]
        
        #Return most common label using numpy
        labels, counts = np.unique(k_labels, return_counts=True)
        return labels[np.argmax(counts)]

csv_file = "Modified_Airlines.csv"  
target_column = "delay_status"  

df = pd.read_csv(csv_file)

df = df.head(10000)  #This is to ensure the code can actually run when memory is low
#IF YOU WOULD LIKE TO RUN THE FULL DATA

#Seperating features (multivariate) and converting to numeric
X = df.drop(columns=[target_column])
y = df[target_column]
X = pd.get_dummies(X, drop_first=True)

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

#Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training and making predictions
knn = KNN(k=3)
knn.fit(X_train_scaled, y_train.values)

y_pred = knn.predict(X_test_scaled)

#CODE BELOW VISUALIZES RESULTS~~~~

#Accuracy cakculations
accuracy = accuracy_score(y_test, y_pred)

#Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Create visuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
from sklearn.metrics import precision_score, recall_score, f1_score
#imported for better visualization

#Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

#Key model metrics in graph form
precision = precision_score(y_test, y_pred, average='binary', pos_label=y_test.unique()[1])
recall = recall_score(y_test, y_pred, average='binary', pos_label=y_test.unique()[1])
f1 = f1_score(y_test, y_pred, average='binary', pos_label=y_test.unique()[1])

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#E6E6FA', '#C8A2C8', '#8F00FF', '#FF00FF']
#Used a random number generator for HEX values that would give

bars = axes[1].barh(metrics, values, color=colors, edgecolor='black')
axes[1].set_xlim(0, 1)
axes[1].set_xlabel('Score', fontsize=12)
axes[1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')

#Add value labels on bars
for bar, value in zip(bars, values):
    axes[1].text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))