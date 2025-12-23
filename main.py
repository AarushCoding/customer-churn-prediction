#Importing libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Loading the data
data = pd.read_csv('telecom_churn.csv')

#Creating the KNN model
model = KNeighborsClassifier(n_neighbors = 5)

#Splitting the data into X and y
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Fitting the model with the training data
model.fit(X_train, y_train)

#Predicting the test data
y_pred = model.predict(X_test)

#Creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Printing the confusion matrix
print(f'Confusion Matrix: \n{cm}')

#Generating an accuracy score
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy:.4f}")
