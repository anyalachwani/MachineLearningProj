from pathlib import Path
import pickle

import numpy as np
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
    grid_search = GridSearchCV(model, grid, cv=5, verbose=2)
    grid_search.fit(inputs_train, target_train)

    best_parameters = grid_search.best_params_
    best_accscore = grid_search.best_score_

    print("Best Parameters:", best_parameters)
    print("Best accuracy on cv holdout data:", best_accscore)

    best_model = grid_search.best_estimator_
    best_model.fit(inputs_train, target_train)

    predictions = best_model.predict(inputs_dev)
    accuracy = accuracy_score(target_dev, predictions)

    print("Accuracy on test data:", accuracy)

    predictions = best_model.predict(inputs_train)
    accuracy = accuracy_score(target_train, predictions)

    print("Accuracy on train data:", accuracy)

    return best_model

if __name__ == '__main__':
    model_savefile = Path('randomforestmodel.pkl')

    if model_savefile.exists():
        print(f'Loading pretrained model from {model_savefile}...')
        best_model = pickle.load(open(model_savefile, 'rb'))
    else:
        print('Training model...')
        best_model = main(RandomForestClassifier(), {
            'n_estimators': [20, 100, 200, 500, 750, 1000],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_depth': [None],
        })

        print('Saving model...')
        with open(model_savefile, 'wb') as f:
            pickle.dump(best_model, f)
    print('Success!')

    rf_importances = best_model.feature_importances_
    feature_names = next(open(
        'heart_failure_clinical_records_dataset.csv'
    )).strip().split(',')[:-1]
    # feature_names and importances were not the same size
    min_length = min(len(rf_importances), len(feature_names))
    rf_importances = rf_importances[:min_length]
    feature_names = feature_names[:min_length]

    # sort
    indices_done = rf_importances.argsort()[::-1]
    importances_done = rf_importances[indices_done]
    feature_names_sorted = np.array(feature_names)[indices_done]

    fig, ax = plt.subplots()
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names_sorted, rotation=45)

    # permutation importance
    pm_importances = permutation_importance(best_model, inputs_train, target_train, n_repeats=5, random_state=30)

    # plot together
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.bar(np.arange(len(feature_names)) - 0.15, importances_done, color='red', label='RF Importances', width=0.3)
    ax.bar(np.arange(len(feature_names)) + 0.15, pm_importances.importances_mean, color='blue', label='Permutation Importance', width=0.3)
    ax.legend()
    plt.show()
