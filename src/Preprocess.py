import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def analyze_data(data):

    # Count rows of data.
    print(data.count())

    # Display the mean age.
    print("Mean Age:", data["age"].mean())

    # Display match rate based on the age of the participant and their gender.
    match_rate_gender = data.groupby(['age', 'gender']).mean()
    print(match_rate_gender['match'])

    # Plot match rate based on age of particpant and their gender.
    plt.figure(figsize=(15, 15))
    match_rate_gender['match'].plot.bar()
    plt.show()

    # Display the match rate when age of participant, age of partner, and gender are considered.
    match_rate_age = data.groupby(['age', 'gender', 'age_o']).mean()
    print(match_rate_age['match'])


def preprocess_field(data):

    processed_data = data.copy()

    # Drop any columns if 23% or more of its values are NaN.
    processed_data = processed_data.dropna(thresh=len(processed_data)*0.77, axis=1)

    # Convert columns to 1-dimensional Series array.
    series = processed_data.columns.to_series()

    # Drop columns to make dataset smaller and more manageable.
    processed_data = processed_data.drop(series["dec"], axis=1)
    processed_data = processed_data.drop(series["iid"], axis=1)
    processed_data = processed_data.drop(series["attr4_1":"amb4_1"], axis=1)
    processed_data = processed_data.drop(series["field"], axis=1)
    processed_data = processed_data.drop(series["gender"], axis=1)
    processed_data = processed_data.drop(series["order"], axis=1)
    processed_data = processed_data.drop(series["fun_o"], axis=1)
    processed_data = processed_data.drop(series["shar_o"], axis=1)
    processed_data = processed_data.drop(series["imprelig"], axis=1)
    processed_data = processed_data.drop(series["prob"], axis=1)
    processed_data = processed_data.drop(series["match_es"], axis=1)
    processed_data = processed_data.drop(series["satis_2"], axis=1)
    processed_data = processed_data.drop(series["amb1_2"], axis=1)
    processed_data = processed_data.drop(series["fun3_2"], axis=1)
    processed_data = processed_data.drop(series["positin1"], axis=1)
    processed_data = processed_data.drop(series["sinc1_2"], axis=1)

    return processed_data


def preprocess_career_c(data):

    # If column 'career_c' is empty in places where column 'career' is not,
    # use column 'career' as the filler value.
    data['career_c'] = data['career_c'].fillna(data['career'])

    # Convert columns to 1-dimensional Series array
    series = data.columns.to_series()

    # Drop the 'career' column after it is used as a filler for column 'career_c'.
    data = data.drop(series["career"], axis=1)

    # Convert all values in column to string type if not already.
    # This will allow us to perform string operations on non-numeric values.
    data['career_c'] = data['career_c'].astype(str)

    # Convert column to all caps so all the data will have the same format.
    data['career_c'] = data['career_c'].str.upper()

    # Remove all spaces between strings in each column.
    data['career_c'] = data['career_c'].str.replace(" ", "")

    # Convert string values to an integer equivalent if the string contains any keywords.
    searchfor = ['LAW']
    data.loc[data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 1

    searchfor = ['RESEARCH']
    data.loc[data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 2

    searchfor = ['ECON']
    data.loc[data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 7

    searchfor = ['HUMAN']
    data.loc[data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 9

    searchfor = ['TECH']
    data.loc[data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 15

    # Convert values back to floating point type.
    # Now there should be no more string values in column 'career_c'.
    data['career_c'] = data['career_c'].astype(float)
    
    return data


def preprocess_sports(data):

    # Convert all values in column to string type if not already.
    # This will allow us to perform string operations on non-numeric values.
    data['sports'] = data['sports'].astype(str)

    # Convert column to all caps so all the data will have the same format.
    data['sports'] = data['sports'].str.upper()

    # Remove all spaces between strings in each column.
    data['sports'] = data['sports'].str.replace(" ", "")

    # Convert string values to NaN if the string contains any keywords.
    searchfor = ['TEACH']
    data.loc[data['sports'].str.contains('|'.join(searchfor), na=False), 'sports'] = np.NaN

    # Convert the column back to floating point format.
    data['sports'] = data['sports'].astype(float)

    # Replace the NaN values that we added with the mean of the column.
    data['sports'] = data['sports'].fillna(data['sports'].mean())
    
    return data


def preprocess_from(data):

    # Change all NaN values to an integer.
    data[data['from'].isnull()] = 0

    # All non-strings are converted to string type.
    data['from'] = data['from'].astype(str)
    
    # Run the Label Encoder algorithm,
    # this will encode all strings to an integer.
    data = encode_data("from", data)

    return data


def preprocess_zipcode(data):

    # Change all NaN values to an integer.
    data[data['zipcode'].isnull()] = 0
    
    # All non-strings are converted to string type.
    data['zipcode'] = data['zipcode'].astype(str)
    
    # Run the Label Encoder algorithm,
    # this will encode all strings to an integer.
    data = encode_data("zipcode", data)

    # All remaining NaN values are converted 
    # to the mean of their respective column.
    data = data.fillna(data.mean())

    return data


def encode_data(target, data):

    # Label Encoder.
    le = LabelEncoder()

    # Convert column to all caps so all the data will have the same format.
    data[target] = data[target].str.upper()

    # Remove all spaces between strings in each column.
    data[target] = data[target].str.replace(" ", "")

    # All string values are encoded to an integer.
    data[target] = le.fit_transform(data[target])

    return data


def redundant_features(target, data):
   
    # Drop the target label, which we save separately.
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Estimator for Recursive Feature Elimination (RFE).
    model = LogisticRegression()
    
    # RFE will use Logistic Regression.
    # The Result will be a unique set of values which represents
    # the column's weight of importance within the dataset.
    rfe = RFE(model, 3)
    fit = rfe.fit(X, y)

    print(fit.n_features_)
    print(fit.support_)
    print(fit.ranking_)

    header_list = list(data.columns.values)

    # Any column with a weight of 88 or over will be removed.
    # We will consider anything above a weight of 88 as redundant.
    ix = 0
    for rank in fit.ranking_:
        if rank >= 88:
            print(header_list[ix])
        ix += 1


def main(_):

    # Read CSV Speed Dating Data using Pandas.
    data = pd.read_csv('../Speed Dating Data.csv')

    # Gives insight about what the data looks like.
    analyze_data(data)
    
    # Preprocess and remove redundant columns.
    processed_data = preprocess_field(data)

    processed_data = preprocess_sports(processed_data)

    # Preprocess the encoded career column.
    processed_data = preprocess_career_c(processed_data)

    # Convert the `from` column to an integer encoding.
    processed_data = preprocess_from(processed_data)

    # Convert the `zipcode` column to an integer encoding.
    # It is easier for the training algorithms to process smaller numbers.
    processed_data = preprocess_zipcode(processed_data)

    # Display redundant features that should be removed.
    target = "match"
    redundant_features(target, processed_data)
    
    # Save dataframe.
    processed_data.to_pickle("dating")

    
if __name__ == '__main__':
    tf.app.run()
