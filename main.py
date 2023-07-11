# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
f = open("/Users/anyalachwani/Downloads/heart_failure_clinical_records_dataset.csv", "r")
data = f.read()
input_data = np.genfromtxt(StringIO(data), delimiter=",", dtype='float64',usecols=np.arange(0,11))
output_data = np.genfromtxt(StringIO(data), delimiter=",", dtype='float64',usecols=(12))
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of test data:", accuracy)

y_pred1 = model.predict(X_train)
accuracy1 = accuracy_score(y_train, y_pred1)
print("Accuracy of training data: ", accuracy1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
