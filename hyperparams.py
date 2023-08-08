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

input_data = np.genfromtxt('heart_failure_clinical_records_dataset.csv', delimiter=',',
                           skip_header=1, dtype='float64', usecols=np.arange(0, 11))
#time_column = np.genfromtxt('heart_failure_clinical_records_dataset.csv', delimiter=',',
 #                           skip_header=1, dtype='float64', usecols=(11))
#input_data = np.column_stack((input_data, time_column))
output_data = np.genfromtxt('heart_failure_clinical_records_dataset.csv', delimiter=',',
                            skip_header=1, dtype='float64', usecols=(12))
inputs_train, inputs_dev, target_train, target_dev = train_test_split(input_data, output_data, test_size=0.2,
                                                                      random_state=42)

# Print the "time" column values
#print("Time Column:", time_column)


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

    with open('randomforestmodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

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
    # importances_done = best_model.feature_importances_

    print("feature importances", best_model.feature_importances_)
    rf_importances = best_model.feature_importances_

    feature_names = next(open(
        'heart_failure_clinical_records_dataset.csv'
    )).strip().split(',')[:-1]
 #   feature_names.append("time")
    print("rf_importance", rf_importances)

    print("feature names", feature_names)
    # remove to test
    # try1 = np.delete(feature_names,-12)
    # try1 = np.delete(feature_names, -10)
    # try1 = feature_names
    # print("try1", try1)
    # feature_names = np.delete(feature_names, -2)

    # print("Length of rf_importances before alignment:", len(rf_importances))
    # print("Length of try1 before alignment:", len(try1))

    # feature_names and importances were not the same size
    # maximum = min(len(rf_importances), len(try1))
    # rf_importances = rf_importances[:maximum]
    # try1 = try1[:maximum]

    # print("Length of rf_importances after alignment:", len(rf_importances))
    # print("Length of try1 after alignment:", len(try1))

    # indices_done = importances_done.argsort()[::-1]
    # importances_done = importances_done[indices_done]
    # feature_names_sorted = np.array(feature_names)[indices_done]

    # sort
    # print("rf_importances before sorting:", rf_importances)
    indices_done = np.argsort(rf_importances)[::-1]
    # print("rf importances after sorting", rf_importances)

    # print("Sorted indices", indices_done)
    importances_done = rf_importances[indices_done]
    feature_names_sorted = np.array(feature_names)[indices_done]
    print("feature names sorted", feature_names_sorted)
    # print(feature_names_sorted)
    # try1_sorted = [try1[i] for i in indices_done]
    # print("try1_sorted", try1_sorted)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names_sorted)), importances_done, tick_label=feature_names_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xticks(range(len(feature_names_sorted)))
    ax.set_xticklabels(feature_names_sorted, rotation=45)

    # permutation importance
    pm_importances = permutation_importance(best_model, inputs_train, target_train, n_repeats=5, random_state=30)
    print(pm_importances)
    # plot together
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.bar(np.arange(len(feature_names_sorted)) - 0.15, importances_done, color='red', label='RF Importances',
           width=0.3)
    ax.bar(np.arange(len(feature_names_sorted)) + 0.15, pm_importances.importances_mean, color='blue',
           label='Permutation Importance', width=0.3, yerr=pm_importances.importances_std)

    ax.legend()
    plt.show()

    # train on each one
    accuracies_test2 = []
    accuracies_train2 = []
    features_changing = []
    for i in range(input_data.shape[1]):
        input_data_altered = np.delete(input_data, i, axis=1)
        xtrain2, xdev2, ytrain2, ydev2 = train_test_split(input_data_altered, output_data, test_size=0.2,
                                                          random_state=42)
        randomforest_one = RandomForestClassifier()
        randomforest_one.fit(xtrain2, ytrain2)
        predictions_train_altered = randomforest_one.predict(xtrain2)
        accuracy_onechanged = accuracy_score(ytrain2, predictions_train_altered)
        accuracies_train2.append(accuracy_onechanged)
        predictions_one1 = randomforest_one.predict(xdev2)
        accuracy_onechanged2 = accuracy_score(ydev2, predictions_one1)
        accuracies_test2.append(accuracy_onechanged2)
        features_changing.append(randomforest_one.feature_importances_)

    # plot importance
    feature_names_1 = next(
        open('heart_failure_clinical_records_dataset.csv')).strip().split(',')[:-1]

    plt.figure(figsize=(10, 6))
    for i, iteration in enumerate(features_changing):
        plt.plot(iteration, label=f"Excluded Feature {i + 1} ({feature_names_1[i]})")

    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importances When Excluding Each Feature")
    plt.xticks(range(len(feature_names_1)), feature_names_1, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

    # accuracy(training)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, input_data.shape[1] + 1), accuracies_train2, marker='o')
    plt.xlabel("Features Excluded")
    plt.ylabel("Accuracy - Training Data")
    plt.title("Features vs. Accuracy")
    plt.xticks(range(1, input_data.shape[1] + 1))
    plt.grid(True)
    plt.show()

    # accuracy(test)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, input_data.shape[1] + 1), accuracies_test2, marker='o')
    plt.xlabel("Features Excluded")
    plt.ylabel("Accuracy - Test Data")
    plt.title("Features vs. Accuracy")
    plt.xticks(range(1, input_data.shape[1] + 1))
    plt.grid(True)
    plt.show()

    # removed time

    # Training 1 changed: 0.702928870292887
    # Training 2 changed: 0.5833333333333334

    # removed

    # train on most important feature
    accuracies_test = []
    accuracies_train = []
    features = []
    indices = np.argsort(rf_importances)[::-1]
    X_train1, X_dev1, y_train1, y_dev1 = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    for i in range(1, len(indices) + 1):
        impfeature_indices = indices[:i]
        most_important_feature_data = input_data[:, impfeature_indices]
        xtrain, xdev = X_train1[:, impfeature_indices], X_dev1[:, impfeature_indices]

        randomforest_mostimportant = RandomForestClassifier()
        randomforest_mostimportant.fit(xtrain, y_train1)
        predictions_mostimportant = randomforest_mostimportant.predict(xtrain)
        accuracy1 = accuracy_score(y_train1, predictions_mostimportant)
        accuracies_train.append(accuracy1)

        predictions_testdata = randomforest_mostimportant.predict(xdev)
        accuracy2 = accuracy_score(y_dev1, predictions_testdata)
        accuracies_test.append(accuracy2)

        features.append(i)

    # On test
    plt.plot(features, accuracies_test, marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy on Test data")
    plt.title(" Number of Features vs Accuracy")
    plt.grid(True)
    plt.show()

    # On train
    plt.plot(features, accuracies_train, marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy on Train data")
    plt.title(" Number of Features vs Accuracy")
    plt.grid(True)
    plt.show()




