import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

input_data = np.genfromtxt('heart_failure_clinical_records_dataset.csv', delimiter=',', skip_header=1, dtype='float64', usecols=np.arange(0, 11))
output_data = np.genfromtxt('heart_failure_clinical_records_dataset.csv', delimiter=',', skip_header=1, dtype='float64', usecols=(12))
inputs_train, inputs_dev, target_train, target_dev = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

@ignore_warnings(category=ConvergenceWarning)
def main(model, grid):
    grid_search = GridSearchCV(model, grid, cv=5, verbose=1)
    grid_search.fit(inputs_train, target_train)

    best_parameters = grid_search.best_params_
    best_accscore = grid_search.best_score_

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_accscore)

    best_model = grid_search.best_estimator_
    best_model.fit(inputs_train, target_train)

    predictions = best_model.predict(inputs_dev)
    accuracy = accuracy_score(target_dev, predictions)

    print("Accuracy on test data:", accuracy)

    predictions = best_model.predict(inputs_train)
    accuracy1 = accuracy_score(target_train, predictions)

    print("Accuracy on train data:", accuracy1)

if __name__ == '__main__':
    main(LogisticRegression(), {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.00001, 0.1, 3.0, 4.0],
        'solver': ['liblinear', 'lbfgs', 'saga', 'LogisticRegression'],
        'max_iter': [100, 200, 300],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced'],
        'tol': [0.0001, 0.001, 0.01],
        'intercept_scaling': [1, 2, 5],
        'multi_class': ['ovr', 'multinomial'],
        'warm_start': [False, True]

    })
    main(MLPClassifier(), {
        'hidden_layer_sizes': [
            (100,), (50,), (100, 100, 100, 100), (50, 50, 50, 50),
            (50, 50, 50, 50, 50), (200,), (200, 200)
        ],
        'batch_size': [15, 30, 50, 90, 100, 150],
        'activation': ['relu', 'tanh', 'sigmoid', 'identity'],
        'max_iter': [100, 200, 300, 500, 1000],
        'learning_rate_init': [0.0001, 0.00001, 0.000001],
        'alpha': [0.001, 0.0001, 0.00001, 0.000001],
    })
    main(RandomForestClassifier(), {
        'n_estimators': [600, 700, 800],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4, 5, 6],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False],
        'max_depth': [None]
    })
