import tensorflow as tf
import numpy as np
import pandas as pd

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

def main(_):

    # Read CSV Speed Dating Data using Pandas.
    data = pd.read_csv('Speed Dating Data.csv')
    
    # Preprocess data.
    processed_data = preprocess(data)
    
    # Save dataframe.
    processed_data.to_pickle("dating")

    
if __name__ == '__main__':
    tf.app.run()
