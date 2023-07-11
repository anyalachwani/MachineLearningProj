import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    filename = 'heart_failure_clinical_records_dataset.csv'
    input_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float64', usecols=np.arange(0, 11))
    output_data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='float64', usecols=(12))
    inputs_train, inputs_dev, target_train, target_dev = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    model = MLPClassifier()
    model.fit(inputs_train, target_train)

    predictions_dev = model.predict(inputs_dev)
    print('Accuracy of test data:', accuracy_score(target_dev, predictions_dev))

    predictions_train = model.predict(inputs_train)
    print('Accuracy of training data: ', accuracy_score(target_train, predictions_train))

if __name__ == '__main__':
    main()
