import tensorflow as tf
import numpy as np
import pandas as pd

def preprocess_field(data):

    processed_data = data.copy()

    # Convert columns to 1-dimensional Series array
    series = processed_data.columns.to_series()

    # Drop columns to make dataset smaller and more manageable.
    processed_data = processed_data.drop(series["undergra":"tuition"], axis=1)
    processed_data = processed_data.drop(series["from":"career"], axis=1)
    processed_data = processed_data.drop(series["sports":], axis=1)
    processed_data = processed_data.drop(series["wave":"positin1"], axis=1)
    processed_data = processed_data.drop(series["iid":"id"], axis=1)
    processed_data = processed_data.drop(series["idg"], axis=1)
    processed_data = processed_data.drop(series["pid"], axis=1)
    processed_data = processed_data.drop(series["field"], axis=1)

    # Drop rows that contain any NaN values.
    processed_data = processed_data.dropna()

    return processed_data


def preprocess_career_c(data):

    processed_data = data.copy()

    # Convert all values in column to string type if not already.
    # This will allow us to perform string operations on non-numeric values.
    processed_data['career_c'] = processed_data['career_c'].astype(str)

    # Convert column to all caps so all the data will have the same format.
    processed_data['career_c'] = processed_data['career_c'].str.upper()

    # Remove all spaces between strings in each column.
    processed_data['career_c'] = processed_data['career_c'].str.replace(" ", "")

    # Convert string values to an integer equivalent if the string contains any keywords.
    searchfor = ['RESEARCH']
    processed_data.loc[processed_data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 2

    searchfor = ['HUMAN']
    processed_data.loc[processed_data['career_c'].str.contains('|'.join(searchfor), na=False), 'career_c'] = 9

    # Convert values back to floating point type.
    # Now there should be no more string values in column 'career_c'.
    processed_data['career_c'] = processed_data['career_c'].astype(float)

    return processed_data


def main(_):

    # Read CSV Speed Dating Data using Pandas.
    data = pd.read_csv('../Speed Dating Data.csv')
    
    # Preprocess field of study column, 
    # as well as remove other redundant columns.
    processed_data = preprocess_field(data)

    # Preprocess the encoded career column.
    processed_data = preprocess_career_c(processed_data)
    
    # Save dataframe.
    processed_data.to_pickle("dating")

    
if __name__ == '__main__':
    tf.app.run()
