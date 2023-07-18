import numpy as np
import sklearn
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

...

f = open("/Users/anyalachwani/Downloads/heart_failure_clinical_records_dataset.csv", "r")
data = f.read()
input_data = np.genfromtxt(StringIO(data), delimiter=",", skip_header=1, dtype='float64', usecols=np.arange(0, 11))
output_data = np.genfromtxt(StringIO(data), delimiter=",", skip_header=1, dtype='float64', usecols=(12))
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)


def func1():
    model = LogisticRegression()

    model_parameters = {

        'penalty': ['l1', 'l2', 'none', 'elasticnet'],
        'C': [0.00001, 0.1, 3.0, 4.0],
        'solver': ['liblinear', 'lbfgs', 'saga', 'LogisticRegression'],
        'max_iter': [100, 200, 300],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced'],
        'tol': [0.0001, 0.001, 0.01],
        'intercept_scaling': [1, 2, 5],
        'multi_class': ['ovr', 'multinomial'],
        'warm_start': [False, True]

    }

    grid_search1 = GridSearchCV(model, model_parameters, cv=10)
    grid_search1.fit(X_train, y_train)

    best_parameters1 = grid_search1.best_params_
    best_accscore1 = grid_search1.best_score_

    print("Best Parameters:", best_parameters1)
    print("Best Score:", best_accscore1)

    best_model = grid_search.best_estimator_
    best_model.fit(inputs_train, target_train)

    predictions = best_model.predict(inputs_dev)
    accuracy = accuracy_score(target_dev, predictions)

    print("Accuracy on test data:", accuracy)

    predictions = best_model.predict(inputs_train)
    accuracy1 = accuracy_score(target_train, predictions)

    print("Accuracy on train data:", accuracy1)


# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy of test data(LogisticRegression):", accuracy)

# y_pred1 = model.predict(X_train)
# accuracy1 = accuracy_score(y_train, y_pred1)
# print("Accuracy of training data(LogisticRegression): ", accuracy1)

@ignore_warnings(category=ConvergenceWarning)
def func2():
    inputs_train, inputs_dev, target_train, target_dev = train_test_split(input_data, output_data, test_size=0.2,
                                                                          random_state=42)
    # model1= MLPClassifier(hidden_layer_sizes=(50), batch_size=15, activation = 'tanh', learning_rate_init=0.01, max_iter=100, solver='adam', alpha=0.001, random_state=42)
    # model1.fit(inputs_train, target_train
    model1 = MLPClassifier()

    # predictions_dev = model1.predict(inputs_dev)
    # print('Accuracy of test data(MLP):', accuracy_score(target_dev, predictions_dev))

    # predictions_train = model1.predict(inputs_train)
    # print('Accuracy of training data(MLP): ', accuracy_score(target_train, predictions_train))

    # Grid to find optimal hyperparameter values
    # grid = {
    # 'hidden_layer_sizes': [(100,), (50,)],
    # 'batch_size': [15, 20, 25, 30, 35, 40],
    # 'activation': ['relu', 'tanh'],
    # 'solver': ['adam'],
    # 'alpha': [0.001, 0.0001],
    # 'learning_rate_init': [0.01, 0.001, 0.1, 0.011],
    # 'max_iter': [100, 200, 300]
    # }
    grid = {
        'hidden_layer_sizes': [(100,), (50,), (100, 100, 100, 100), (50, 50, 50, 50), (50, 50, 50, 50, 50), (200,),
                               (200, 200)],
        'batch_size': [15, 30, 50, 90, 100, 150],
        'activation': ['relu', 'tanh', 'sigmoid', 'identity'],
        'max_iter': [100, 200, 300, 500, 1000],
        'learning_rate_init': [0.0001, 0.00001, 0.000001],
        'alpha': [0.001, 0.0001, 0.00001, 0.000001],

    }

    grid_search = GridSearchCV(model1, grid, cv=9)
    grid_search.fit(inputs_train, target_train)

    best_parameters = grid_search.best_params_
    best_accscore = grid_search.best_score_

    # Print the best parameters and best score
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


# best_model = MLPClassifier(**grid_search.best_params_)
# best_model.fit(inputs_train, target_train)
# predictions_val = best_model.predict(inputs_train)
# accuracy_val = accuracy_score(target_train, predictions_val)
# print("Accuracy on validation data:", accuracy_val)

# predictions_test = best_model.predict(inputs_test)
# accuracy_test = accuracy_score(target_dev, predictions_test)
# print("Accuracy on test data:", accuracy_test)


def main_function():
    # 85% training accuracy 70% dev
    from sklearn.ensemble import RandomForestClassifier
    inputs_train, inputs_dev, target_train, target_dev = train_test_split(input_data, output_data, test_size=0.2,
                                                                          random_state=42)
    model3 = RandomForestClassifier()
    grid = {
        'n_estimators': [600, 700, 800],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4, 5, 6],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False],
        'max_depth': [None]

    }

    grid_search = GridSearchCV(model3, grid, cv=5)
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


func1()
func2()
main_function()
