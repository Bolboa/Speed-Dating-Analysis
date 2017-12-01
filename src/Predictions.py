import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
from sklearn import datasets, svm, model_selection, tree, metrics, preprocessing


def decision_tree(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Split data in a 20/80 split.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Initialize a decison tree with a max_depth of 10.
    clf_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

    # Fit the model with the training data.
    clf_tree.fit(X_train, y_train)

    # Use the test data to calculate the score.
    print("Decision Tree Score (No Cross-validation):", clf_tree.score(X_test, y_test))

    # Implement the decision tree again using Cross Validation.
    unique_permutations_cross_val(X, y, clf_tree)

    return clf_tree


def random_forest(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Random Forest Classifier.
    clf_tree = ske.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

    # Implement the decision tree again using Cross Validation.
    unique_permutations_cross_val(X, y, clf_tree)

    clf_tree.fit(X, y)
    
    return clf_tree


def gradient_boosting(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Gradient Boosting.
    clf_gradient = ske.GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)

    # Implement the decision tree again using Cross Validation.
    unique_permutations_cross_val(X, y, clf_gradient)

    clf_gradient.fit(X, y)
    
    return clf_gradient


def unique_permutations_cross_val(X, y, model):

    # Split data 20/80 to be used in a K-Fold Cross Validation with unique permutations.
    shuffle_validator = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    
    # Calculate the score of the model after Cross Validation has been applied to it. 
    scores = model_selection.cross_val_score(model, X, y, cv=shuffle_validator)

    # Print out the score (mean), as well as the variance.
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))


def main(_):

    # Read dataframe.
    data = pd.read_pickle('dating')
    
    # Set target column.
    target = "match"

    # Extract all column headers.
    header_list = list(data.columns.values)

    # Get the mean values for all columns and store them in a list.
    mean_values = [data[header].mean() for header in header_list]

    # Map each column header to its respective mean.
    header_dict = dict(zip(header_list, mean_values))

    # Remove the target header as this is not part of the dataset.
    del header_dict[target]

    # Key = Column Header,
    # Value = Mean of Column,
    # Storing it in a dictionary makes it easier to change values so
    # as to see how the prediction changes.
    header_dict['exercise'] = 10
    header_dict['age'] = 23
    header_dict['movies'] = 10
    header_dict['reading'] = 10
    header_dict['music'] = 10

    # Extract only the mean values after values have been changed.
    extract_values = list(header_dict.values())
    
    # Decision Tree
    decision_tree = decision_tree(target, data)
    
    # Random Forest
    rnd_forest = random_forest(target, data)
    
    # Gradient Boosting
    grd_boost = gradient_boosting(target, data)    
    
    # Random Forest can predict the match rate if you give a value for every column.
    # In this case the values represent the mean of every column.
    predictions = rnd_forest.predict([extract_values])
    print(predictions)


if __name__ == '__main__':
    tf.app.run()
