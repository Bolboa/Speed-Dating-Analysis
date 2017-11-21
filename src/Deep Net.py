import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 20, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 20, 'Batch size. Must divide evenly into the data set sizes.')
flags.DEFINE_integer('display_step', 2, 'Tells function to print out progress after every epoch')


def DNN(target, data):

	# Drop the target label, which we save separately.
    X = np.array(data.drop([target], axis=1).values)
    Y = np.array(data[target].values)
    Y = np.reshape(Y, (-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(
    	X, 
    	Y, 
    	test_size=0.2
    )

    rng = np.random

    n_rows = X_train.shape[0]

    X = tf.placeholder("float")
    Y = tf.placeholder("float")


    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    pred = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_sum(tf.pow(pred-Y, 2)/(2*n_rows))

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    with tf.Session() as sess:

    	sess.run(init)

    	for epoch in range(FLAGS.training_epochs):

    		avg_cost = 0

    		for (x, y) in zip(X_train, Y_train):

    			sess.run(optimizer, feed_dict={X:x, Y:y})

    		# display logs per epoch step
    		if (epoch + 1) % FLAGS.display_step == 0:

    			c = sess.run(
    				cost, 
    				feed_dict={X:X_train, Y:Y_train}
    			)

    			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    	print("Optimization Finished!")

    	training_cost = sess.run(cost, feed_dict={X:X_train, Y:Y_train})
    	print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    	testing_cost = sess.run(
    		tf.reduce_sum(tf.pow(pred-Y, 2)) / (2*X_test.shape[0]),
    		feed_dict={X:X_test, Y:Y_test}
    	)

    	print("Testing cost=", testing_cost)
    	print("Absolute mean square loss difference:", abs(
    		training_cost - testing_cost
    	))


def main(_):

	# Read dataframe.
    data = pd.read_pickle('dating')

    target = 'match'

    DNN(target, data)

    
if __name__ == '__main__':
    tf.app.run()
