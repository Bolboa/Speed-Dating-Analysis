import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
from sklearn import datasets, svm, model_selection, tree, metrics, preprocessing


def analyze_data(data):

    # Count rows of data.
    print(data.count())

    # Display the mean age.
    print(data["age"].mean())

    # Display match rate based on the age of the participant and their gender.
    match_rate_gender = data.groupby(['age', 'gender']).mean()

    # Plot match rate based on age of particpant and their gender.
    match_rate_gender['match'].plot.bar()
    plt.show()

    # Display the match rate when age of participant, age of partner, and gender are considered.
    match_rate_age = data.groupby(['age', 'gender', 'age_o']).mean()
    print(match_rate_age['match'])

def decision_tree(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Split data in a 20/80 split.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Initialize a decison tree with a max_depth of 10.
    clf_tree = tree.DecisionTreeClassifier(max_depth=10)

    # Fit the model with the training data.
    clf_tree.fit(X_train, y_train)

    # Use the test data to calculate the score.
    print(clf_tree.score(X_test, y_test))

    # Implement the decision tree again using Cross Validation.
    unique_permutations_cross_val(X, y, clf_tree)

    return clf_tree

def random_forest(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Run Cross Validation on Random Forest Classifier.
    clf_tree = ske.RandomForestClassifier(n_estimators=50)
    unique_permutations_cross_val(X, y, clf_tree)

    clf_tree.fit(X, y)
    return clf_tree


def gradient_boosting(target, data):

    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Run Cross Validation on Gradient Boosting.
    clf_gradient = ske.GradientBoostingClassifier(n_estimators=50)
    unique_permutations_cross_val(X, y, clf_gradient)


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


    gender      = 0
    condtn      = 1
    order       = 5
    partner     = 1
    match       = 1
    int_corr    = 0.30
    samerace    = 1
    age_o       = 21
    race_o      = 1
    pf_o_att    = 20
    pf_o_sin    = 20    
    pf_o_int    = 20
    pf_o_fun    = 20
    pf_o_amb    = 20
    pf_o_sha    = 20
    attr_o      = 5
    sinc_o      = 5
    intel_o     = 5 
    fun_o       = 5
    amb_o       = 5
    shar_o      = 5
    like_o      = 5
    prob_o      = 5
    met_o       = 5
    age         = 21
    field_cd    = 3
    race        = 1
    imprace     = 4
    imprelig    = 4
    career_c    = 2


    #analyze_data(data)
    #decision_tre = decision_tree(target, data)
    rnd_forest = random_forest(target, data)
    #gradient_boosting(target, data)

    print(rnd_forest.predict([[
        gender, 
        condtn,
        order,
        partner,
        match,
        int_corr,
        samerace,
        age_o,
        race_o,
        pf_o_att,
        pf_o_sin,
        pf_o_int,
        pf_o_fun,
        pf_o_amb,
        pf_o_sha,
        attr_o,
        sinc_o,
        intel_o,
        fun_o,
        amb_o,
        shar_o,
        like_o,
        prob_o,
        met_o,
        age,
        field_cd,
        race,
        imprace,
        imprelig,
        career_c
    ]]))
    
    
if __name__ == '__main__':
    tf.app.run()
