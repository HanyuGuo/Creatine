'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 2000 # 1st layer number of features
n_hidden_2 = 2500 # 2nd layer number of features
n_hidden_3 = 3000 # 1st layer number of features
n_hidden_4 = 4000 # 2nd layer number of features
n_hidden_5 = 4000 # 1st layer number of features
n_hidden_6 = 3000# 2nd layer number of features
n_hidden_7 = 2500 # 1st layer number of features
n_hidden_8 = 2000 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    # Hidden layer with RELU activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)
    # Hidden layer with RELU activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    # Hidden layer with RELU activation
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
    'out': tf.Variable(tf.random_normal([n_hidden_8, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'b8': tf.Variable(tf.random_normal([n_hidden_8])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                      y: batch_ys})
        # print (c)
    # print ("-------------------------------------")
    #     # Compute average loss
        avg_cost += c / total_batch
    # # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", np.sum(avg_cost))

print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


import time
bs = 20
start = time.time()
for i in range(mnist.test.images.shape[0]/bs):
    print("Accuracy:", sess.run(accuracy, {x: mnist.test.images[i*bs:(i+1)*bs], y: mnist.test.labels[i*bs:(i+1)*bs]}))
end = time.time()
print(end - start)


# sess.run(logits, {x: mnist.test.images[:5], y: mnist.test.labels[:5]})






ww1 = sess.run(weights['h1'])
ww2 = sess.run(weights['h2'])
ww3 = sess.run(weights['h3'])
ww4 = sess.run(weights['h4'])
ww5 = sess.run(weights['h5'])
ww6 = sess.run(weights['h6'])
ww7 = sess.run(weights['h7'])
ww8 = sess.run(weights['h8'])
ww9 = sess.run(weights['out'])

bb1 = sess.run(biases['b1'])
bb2 = sess.run(biases['b2'])
bb3 = sess.run(biases['b3'])
bb4 = sess.run(biases['b4'])
bb5 = sess.run(biases['b5'])
bb6 = sess.run(biases['b6'])
bb7 = sess.run(biases['b7'])
bb8 = sess.run(biases['b8'])
bb9 = sess.run(biases['out'])


np.savetxt("./f_h1.txt", ww1.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h2.txt", ww2.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h3.txt", ww3.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h4.txt", ww4.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h5.txt", ww5.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h6.txt", ww6.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h7.txt", ww7.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_h8.txt", ww8.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_hout.txt", ww9.reshape(-1), delimiter=',', newline=',')

np.savetxt("./f_b1.txt", bb1.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b2.txt", bb2.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b3.txt", bb3.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b4.txt", bb4.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b5.txt", bb5.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b6.txt", bb6.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b7.txt", bb7.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_b8.txt", bb8.reshape(-1), delimiter=',', newline=',')
np.savetxt("./f_bout.txt", bb9.reshape(-1), delimiter=',', newline=',')




