import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 20, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 20, 'Batch size. Must divide evenly into the data set sizes.')
flags.DEFINE_integer('display_step', 2, 'Tells function to print out progress after every epoch')


def DNN(target, data):

	# Drop the target label, which we save separately.
    X_data = np.array(data.drop([target], axis=1).values)
    Y_data = np.array(data[target].values)
    Y_data = np.reshape(Y_data, (-1, 1))

    # Split the data into testing sample and training sample at 80/20.
    X_train, X_test, Y_train, Y_test = train_test_split(
    	X_data, 
    	Y_data, 
    	test_size=0.2
    )

    # Number of rows.
    n_rows = X_train.shape[0]

    X = tf.placeholder(tf.float32, [None, 89])
    Y = tf.placeholder(tf.float32, [None, 1])

    # Tensor size is defined to calculate weight and bias.
    W_shape = tf.TensorShape([89, 1])
    b_shape = tf.TensorShape([1])

    # Tensor size is used and random values are used to fill up the variables.
    # It does not matter what values we start with since Linear Regression models
    # will always end up at a global minimum through training.
    W = tf.Variable(tf.random_normal(W_shape))
    b = tf.Variable(tf.random_normal(b_shape))

    # We use matmul because we are training on a multi-class dataset.
    pred = tf.add(tf.matmul(X, W), b)

    # Define the cost function.
    cost = tf.reduce_sum(tf.pow(pred-Y, 2)/(2*n_rows-1))

    # Minimize cost through gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    # Initialize all global variables.
    init = tf.global_variables_initializer()
    
    # Run session.
    with tf.Session() as sess:

    	sess.run(init)

    	for epoch in range(FLAGS.training_epochs):

    		# Loop through the training sample.
    		for (x, y) in zip(X_train, Y_train):

    			# Batch size is 1 since we loop through every single one iteratively.
    			x = np.reshape(x, (1, 89))
    			y = np.reshape(y, (1,1))

    			# Run gradient descent to minimize cost.
    			sess.run(optimizer, feed_dict={X:x, Y:y})

    		# Display logs per epoch depending on the display step.
    		if (epoch + 1) % FLAGS.display_step == 0:

    			# Calculate cost.
    			c = sess.run(
    				cost, 
    				feed_dict={X:X_train, Y:Y_train}
    			)

    			# R-squared score checks how close the test data is
    			# to the fitted line.
    			Y_pred = sess.run(pred, feed_dict={X:X_test})
    			test_error = r2_score(Y_test, Y_pred)
    			print(test_error)

    			# Print cost per epoch
    			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    	print("Optimization Finished!")

    	# Get all predictions for the test sample.
    	Y_pred = sess.run(pred, feed_dict={X:X_test})
    	
    	# Calculate how far off each prediction is from the target label,
    	# we use the definition of Mean Squared Error to calculate this.
    	mse = tf.reduce_mean(tf.square(Y_pred - Y_test))
    	print("MSE (definition): %4f" % sess.run(mse))

    	# Cast test sample to float.
    	Y_test = tf.cast(Y_test, tf.float32)
    	
    	# Use Tensorflow's definition of a Mean Squared Error.
    	mse = tf.metrics.mean_squared_error(labels=Y_test, predictions=Y_pred)

    	# Intialize local variables to be used to calculate the Mean Square Error.
    	init_local = tf.local_variables_initializer()
    	sess.run(init_local)

    	# Calculate the Mean Square Error.
    	mse = sess.run(mse)
    	print("MSE: %4f" % mse[1])

    	# Display how the fitted lines changes as more data is trained on.
    	plt.plot(X_train, Y_train, 'ro', label='main')
    	plt.plot(X_train, X_train*sess.run(W).T, label='Predicted')
    	plt.show()


def main(_):

	# Read dataframe.
    data = pd.read_pickle('dating')

    target = 'match'

    DNN(target, data)

    
if __name__ == '__main__':
    tf.app.run()
