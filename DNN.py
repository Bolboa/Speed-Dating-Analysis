import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd

def preprocess():

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

    data['field'] = data['field'].str.replace(" ", "")

    # Convert string values to an integer equivalent if the string contains any keywords.
    searchfor = ['LAW']
    data.loc[data['field'].str.contains('|'.join(searchfor)), 'field'] = 1

    searchfor = ['MATH', 'STAT']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 2

    searchfor = ['PSYCHOLOG', 'SOCIALSCIENCE', 'ANTHROPOLOGY', 'HUMAN', 'PATHOLOGY', 'EPIDEM']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 3

    searchfor = ['BIOTECH', 'PHARM', 'MED', 'NEURO']   
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 4

    searchfor = ['ENGINEER', 'COMPUTERSCIENCE', 'ENGG']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 5

    searchfor = ['ENGLISH', 'LITERATURE', 'WRITING', 'CREATIVE', 'JOURNAL']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 6

    searchfor = ['HIST', 'RELIGION', 'PHIL', 'CLASSICS']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 7

    searchfor = ['BUSINESS', 'ECON', 'FINANCE', 'MBA', 'MARKETING', 'FINANACE', 'MONEY']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 8

    searchfor = ['EDUCATION', 'ACADEMIA']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 9

    searchfor = ['BIO', 'CHEM', 'PHYSIC', 'GENETICS', 'ECOLOGY']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 10

    searchfor = ['SOCIALWORK', 'SOCIALSTUDIES', 'SOCIOLOGY']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 11

    searchfor = ['UNDER', 'UNDECIDED']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 12

    searchfor = ['POLI', 'NATIONAL']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 13

    searchfor = ['FILM']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 14

    searchfor = ['ARTS', 'ADMINISTRATION', 'ACTING', 'MFA', 'THEATER', 'THEATRE']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 15

    searchfor = ['LANGUAGES', 'FRENCH']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 16

    searchfor = ['ARCHITECT']
    data.loc[data['field'].str.contains('|'.join(searchfor), na=False), 'field'] = 17

    # Most columns are now integer values,
    # if there are any values that were not converted then we label those as Other = 18.
    totalsearch = [i for i in range(1,18)]
    data.loc[~data['field'].isin(totalsearch), 'field'] = 18

    return data

def deep_net(label, data):

    # Target label used for training
    labels = np.array(data[label], dtype=np.float32)
    
    # Reshape target label from (6605,) to (6605, 1)
    labels = np.reshape(labels, (-1, 1))
   
    # Data for training minus the target label.
    data = np.array(data.drop(label, axis=1), dtype=np.float32)
    
    # Deep Neural Network.    
    net = tflearn.input_data(shape=[None, 32])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 1, activation='softmax')
    net = tflearn.regression(net)

    # Define model.
    model = tflearn.DNN(net)
    model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
    

def main(_):
    
    data = preprocess()
    label = "age"
    deep_net(label, data)
    
if __name__ == '__main__':
    tf.app.run()
