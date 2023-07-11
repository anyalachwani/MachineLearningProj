import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    filename = 'heart_failure_clinical_records_dataset.csv'
    input_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float64', usecols=np.arange(0, 11))
    output_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float64', usecols=(12))
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    print('Accuracy of test data:', accuracy)

    train_pred = model.predict(X_train)
    accuracy1 = accuracy_score(y_train, train_pred)
    print('Accuracy of training data: ', accuracy1)

if __name__ == '__main__':
    main()
