import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

    return best_model

if __name__ == '__main__':
    # main(LogisticRegression(), {
    #   'penalty': ['l1', 'l2', 'elasticnet'],
    #  'C': [0.00001, 0.1, 3.0, 4.0],
    # 'solver': ['liblinear', 'lbfgs', 'saga', 'LogisticRegression'],
    # 'max_iter': [100, 200, 300],
    # 'fit_intercept': [True, False],
    # 'class_weight': [None, 'balanced'],
    # 'tol': [0.0001, 0.001, 0.01],
    # 'intercept_scaling': [1, 2, 5],
    # 'multi_class': ['ovr', 'multinomial'],
    # 'warm_start': [False, True]

    # })
    # main(MLPClassifier(), {
    #   'hidden_layer_sizes': [
    #      (100,), (50,), (100, 100, 100, 100), (50, 50, 50, 50),
    #     (50, 50, 50, 50, 50), (200,), (200, 200)
    # ],
    # 'batch_size': [15, 30, 50, 90, 100, 150],
    # 'activation': ['relu', 'tanh', 'sigmoid', 'identity'],
    # 'max_iter': [100, 200, 300, 500, 1000],
    # 'learning_rate_init': [0.0001, 0.00001, 0.000001],
    # 'alpha': [0.001, 0.0001, 0.00001, 0.000001],
    # })
    best_model = main(RandomForestClassifier(), {

        'n_estimators': [20, 100, 200, 500, 750, 1000],
        # 'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_depth': [100],
        'max_features': ['auto', 'sqrt', 'log2'],

        # 'n_estimators': [100, 200, 300, 400, 500, 1000], #increases number of trees
        # 'min_samples_split': [10, 20, 30], # higher the value, simpler the tree; outdoes max_depth, number of nodes to split tree
        # 'min_samples_leaf': [1, 2, 3, 4, 8], #also controls overfitting, number of nodes to create leaf
        # 'max_features': ['auto', 'sqrt'], #number of features for split
        # 'bootstrap': [True, False], #if true, bagging occurs. if False, full data set used to train each tree
        # 'max_depth': [10, 50, 200], #fit training data better; depth of the decision trees
        # 'class_weight': [None, "balanced"] # different weights to classes - useful when there is imbalance in the data set(precaution)
    })

    with open('randomforestmodel.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    with open('randomforestmodel.pkl', 'rb') as f:
        randomforest_model = pickle.load(f)

    importances = randomforest_model.feature_importances_
    feature_names = next(open(
        'heart_failure_clinical_records_dataset.csv'
    )).strip().split(',')[:-1]
    # feature_names and importances were not the same size
    min_length = min(len(importances), len(feature_names))
    importances = importances[:min_length]
    feature_names = feature_names[:min_length]

    # sort
    indices_done = importances.argsort()[::-1]
    importances_done = importances[indices_done]
    feature_names_sorted = np.array(feature_names)[indices_done]

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(feature_names)), importances_done, color='red', label='RF Importances')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names_sorted, rotation=45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.legend()
    plt.show()

    # permutation importance
    importances_permuationimportances = permutation_importance(best_model, inputs_train, target_train, n_repeats=5, random_state=30)
    importances1 = importances_permuationimportances.importances_mean
    importances_final = importances_permuationimportances.importances_std

    # plot together
    ax.bar(np.arange(len(feature_names)) - 0.15, importances_done, color='red', label='RF Importances', width=0.3)
    ax.bar(np.arange(len(feature_names)) + 0.15, importances_permuationimportances, color='blue', label='Permutation Importance', width=0.3)

# Test 1 for Random Forest Classifier

# Best Parameters: {'class_weight': None, 'max_depth': 50, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 500}
# n_estimators, min_samples_leaf, max_features, max_depth, class_weight
# Acc Score: 76.9, Training: 89.4,  Testing: 73.3


# Test 2 Random Forest Classifier
# max_depth, max_features, min_samples_leaf, n_estimators
# Best Parameters: {'max_depth': 80, 'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 400}
# Best Score: 76.5%, train: 81%, test: 67%


# Test 3 RFC
# Best Parameters: {'bootstrap': True, 'max_depth': 200, 'min_samples_leaf': 2, 'n_estimators': 500}
# bootstrap, max_depth, min_samples_leaf, n_estimators
# Best Score: 0.7692375886524824, Accuracy on test data: 0.7, Accuracy on train data: 0.9831932773109243


# Best Parameters: {'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'n_estimators': 200}
# Best Score: 0.7733156028368795
# Accuracy on test data: 0.7166666666666667
# Accuracy on train data: 0.9159663865546218
