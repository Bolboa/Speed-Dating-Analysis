import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd

def preprocess(cols_del):

    # Read CSV Speed Dating Data using Pandas
    data = pd.read_csv('/home/eric/Documents/Speed Dating Data.csv')

    # Convert columns to 1-dimensional Series array
    series = data.columns.to_series()

    # Drop columns to make dataset smaller and more manageable.
    data = data.drop(series["field_cd":], axis=1)
    data = data.drop(series["position":"positin1"], axis=1)

    # Drop rows that contain any NaN values.
    data = data.dropna()

    # Convert column to all caps so all the data will have the same format.
    data['field'] = data['field'].str.upper()

    # Convert string values to an integer equivalent if the string contains any keywords.
    searchfor = ['LAW']
    data.loc[data['field'].str.contains('|'.join(searchfor)), 'field'] = 1

    searchfor = ['MATH']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 2

    searchfor = ['PSYCHOLOG', 'SOCIAL SCIENCE']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 3

    searchfor = ['BIO TECH', 'PHARM', 'MED']   
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 4

    searchfor = ['ENGINEER']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 5

    searchfor = ['ENGLISH', 'LITERATURE', 'WRITING', 'CREATIVE', 'JOURNAL']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 6

    searchfor = ['HIST', 'RELIGION', 'PHIL']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 7

    searchfor = ['BUSINESS', 'ECON', 'FINANCE']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 8

    searchfor = ['EDUCATION', 'ACADEMIA']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 9

    searchfor = ['BIO', 'CHEM', 'PHYSIC']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 10

    searchfor = ['SOCIAL WORK']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 11

    searchfor = ['UNDER', 'UNDECIDED']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 12

    searchfor = ['POLI', 'INTERNAT']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 13

    searchfor = ['FILM']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 14

    searchfor = ['ARTS']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 15

    searchfor = ['LANGUAGES']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 16

    searchfor = ['ARCHITECT']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 17

    searchfor = ['OTHER']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 18

    print(data)

def main(_):
    delete = [1]
    preprocess(delete)
if __name__ == '__main__':
    tf.app.run()
