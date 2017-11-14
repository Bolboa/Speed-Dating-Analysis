import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
from sklearn import datasets, svm, model_selection, tree, metrics

def example():
    
    data = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    print(data.shape)
    # Target label used for training
    labels = np.array(data['A'].values, dtype=np.float32)
    
    # Reshape target label from (6605,) to (6605, 1)
    labels = np.reshape(labels, (-1, 1))
    
    # Data for training minus the target label.
    data = np.array(data.drop('A', axis=1).values, dtype=np.float32)

    # Deep Neural Network.    
    net = tflearn.input_data(shape=[None, 3])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 1, activation='softmax')
    net = tflearn.regression(net)

    # Define model.
    model = tflearn.DNN(net)
    model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)



def preprocess(data):

    processed_data = data.copy()

    # Convert columns to 1-dimensional Series array
    series = processed_data.columns.to_series()

    # Drop columns to make dataset smaller and more manageable.
    processed_data = processed_data.drop(series["field_cd":], axis=1)
    processed_data = processed_data.drop(series["position":"positin1"], axis=1)
    processed_data = processed_data.drop(series["iid":"id"], axis=1)
    processed_data = processed_data.drop(series["idg"], axis=1)

    # Drop rows that contain any NaN values.
    processed_data = processed_data.dropna()

    # Convert column to all caps so all the data will have the same format.
    processed_data['field'] = processed_data['field'].str.upper()

    processed_data['field'] = processed_data['field'].str.replace(" ", "")

    # Convert string values to an integer equivalent if the string contains any keywords.
    searchfor = ['LAW']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor)), 'field'] = 1

    searchfor = ['MATH', 'STAT']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 2

    searchfor = ['PSYCHOLOG', 'SOCIALSCIENCE', 'ANTHROPOLOGY', 'HUMAN', 'PATHOLOGY', 'EPIDEM']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 3

    searchfor = ['BIOTECH', 'PHARM', 'MED', 'NEURO']   
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 4

    searchfor = ['ENGINEER', 'COMPUTERSCIENCE', 'ENGG']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 5

    searchfor = ['ENGLISH', 'LITERATURE', 'WRITING', 'CREATIVE', 'JOURNAL']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 6

    searchfor = ['HIST', 'RELIGION', 'PHIL', 'CLASSICS']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 7

    searchfor = ['BUSINESS', 'ECON', 'FINANCE', 'MBA', 'MARKETING', 'FINANACE', 'MONEY']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 8

    searchfor = ['EDUCATION', 'ACADEMIA']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 9

    searchfor = ['BIO', 'CHEM', 'PHYSIC', 'GENETICS', 'ECOLOGY']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 10

    searchfor = ['SOCIALWORK', 'SOCIALSTUDIES', 'SOCIOLOGY']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 11

    searchfor = ['UNDER', 'UNDECIDED']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 12

    searchfor = ['POLI', 'NATIONAL']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 13

    searchfor = ['FILM']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 14

    searchfor = ['ARTS', 'ADMINISTRATION', 'ACTING', 'MFA', 'THEATER', 'THEATRE']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 15

    searchfor = ['LANGUAGES', 'FRENCH']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 16

    searchfor = ['ARCHITECT']
    processed_data.loc[processed_data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 17

    # Most columns are now integer values,
    # if there are any values that were not converted then we label those as Other = 18.
    totalsearch = [i for i in range(1,18)]
    processed_data.loc[~processed_data['field'].isin(totalsearch), 'field'] = 18

    return processed_data

def deep_net(label, data):

    # Target label used for training
    labels = np.array(data[label].values, dtype=np.float32)
    
    # Reshape target label from (6605,) to (6605, 1)
    labels = np.reshape(labels, (-1, 1))
    
    # Data for training minus the target label.
    data = np.array(data.drop(label, axis=1).values, dtype=np.float32)

    print(data.shape)
    print(labels.shape)
    # Deep Neural Network.    
    net = tflearn.input_data(shape=[None, 29])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 1, activation='softmax')
    net = tflearn.regression(net)

    # Define model.
    model = tflearn.DNN(net)
    model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
    

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

def decision_tree(data):

    # Drop the target label, which we save separately.
    X = data.drop(['match'], axis=1).values
    y = data['match'].values

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


def unique_permutations_cross_val(X, y, model):

    # Split data 20/80 to be used in a K-Fold Cross Validation with unique permutations.
    shuffle_validator = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    
    # Calculate the score of the model after Cross Validation has been applied to it. 
    scores = model_selection.cross_val_score(model, X, y, cv=shuffle_validator)

    # Print out the score (mean), as well as the variance.
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))


def random_forest(data):

    # Drop the target label, which we save separately.
    X = data.drop(['match'], axis=1).values
    y = data['match'].values

    clf_tree = ske.RandomForestClassifier(n_estimators=50)
    unique_permutations_cross_val(X, y, clf_tree)


def main(_):

    # Read CSV Speed Dating Data using Pandas
    data = pd.read_csv('/home/eric/Documents/Speed Dating Data.csv')
    
    # Preprocess data.
    processed_data = preprocess(data)
    
    #label = "age"
    #deep_net(label, processed_data)
    #analyze_data(processed_data)
    #decision_tree(processed_data)
    random_forest(processed_data)
    #example()
    
if __name__ == '__main__':
    tf.app.run()
