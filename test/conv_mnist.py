'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return x


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 2)
    conv1_relu = tf.nn.relu(conv1)
    # Convolution Layer
    conv2 = conv2d(conv1_relu, weights['wc2'], biases['bc2'], 2)
    conv2_relu = tf.nn.relu(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2_relu, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out, conv1, conv2, fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 5])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 5, 10])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*10, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred, conv1, conv2, fc1 = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)
step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                   keep_prob: dropout})
    if step % display_step == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 1.})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 256 mnist test images
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                  y: mnist.test.labels[:256],
                                  keep_prob: 1.}))


# ppred = sess.run(pred, feed_dict={x: mnist.test.images[:2],
#                                   y: mnist.test.labels[:2],
#                                   keep_prob: 1.})
# cconv1 = sess.run(conv1, feed_dict={x: mnist.test.images[:2],
#                                   y: mnist.test.labels[:2],
#                                   keep_prob: 1.})
# cconv2 = sess.run(conv2, feed_dict={x: mnist.test.images[:2],
#                                   y: mnist.test.labels[:2],
#                                   keep_prob: 1.})
# ffc1 = sess.run(fc1, feed_dict={x: mnist.test.images[:2],
#                                   y: mnist.test.labels[:2],
#                                   keep_prob: 1.})

cconv1 = sess.run(conv1, feed_dict={x: mnist.test.images[:1],
                                  y: mnist.test.labels[:1],
                                  keep_prob: 1.})


wc1 = sess.run(weights['wc1'])
wc2 = sess.run(weights['wc2'])
wd1 = sess.run(weights['wd1'])
wout = sess.run(weights['out'])

bc1 = sess.run(biases['bc1'])
bc2 = sess.run(biases['bc2'])
bd1 = sess.run(biases['bd1'])
bout = sess.run(biases['out'])


np.savetxt("../data/cconv1.txt", cconv1.reshape(-1), delimiter=',', newline=',')

np.savetxt("../data/wc1.txt", wc1.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/wc2.txt", wc2.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/wd1.txt", wd1.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/wout.txt", wout.reshape(-1), delimiter=',', newline=',')

np.savetxt("../data/bc1.txt", bc1.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/bc2.txt", bc2.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/bd1.txt", bd1.reshape(-1), delimiter=',', newline=',')
np.savetxt("../data/bout.txt", bout.reshape(-1), delimiter=',', newline=',')


import tensorflow as tf
import numpy as np
sess = tf.Session()

data1 = np.array([1,2,3,4,5,6,7,8,9,2,2,2,4,5,6,7,8,9,3,2,1,6,5,4,9,8,7,
                  5,7,8,9,3,4,2,3,4,7,1,5,7,1,5,7,3,5,7,7,1,8,7,1,3,5,1], dtype=np.float32)
data1 = data1.reshape(2,3,3,3)

ft1 = np.array([1,2.0,3,4,5,6,7,4,5,3,1,2,5,7,8,6,7,8,9,5,1,2,4,8], dtype=np.float32)
ft1 = ft1.reshape(2,2,3,2)


data2 = np.array([1,2,3,4], dtype=np.float32)
data2 = data2.reshape(1,2,2,1)

ft2 = np.array([1,2.0,3,4], dtype=np.float32)
ft2 = ft2.reshape(2,2,1,1)


data3 = np.array([1,2,3,4,7,8,9,11], dtype=np.float32)
data3 = data3.reshape(1,2,2,2)

ft3 = np.array([1,2.0,3,4,2,2,3,8], dtype=np.float32)
ft3 = ft3.reshape(2,2,2,1)

real_data = np.array([0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.32941177,  0.72549021,  0.62352943,
         0.59215689,  0.23529413,  0.14117648,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.8705883 ,  0.99607849,  0.99607849,  0.99607849,  0.99607849,
         0.9450981 ,  0.77647066,  0.77647066,  0.77647066,  0.77647066,
         0.77647066,  0.77647066,  0.77647066,  0.77647066,  0.66666669,
         0.20392159,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.26274511,  0.44705886,
         0.28235295,  0.44705886,  0.63921571,  0.89019614,  0.99607849,
         0.88235301,  0.99607849,  0.99607849,  0.99607849,  0.98039222,
         0.89803928,  0.99607849,  0.99607849,  0.54901963,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.06666667,  0.25882354,  0.05490196,  0.26274511,
         0.26274511,  0.26274511,  0.23137257,  0.08235294,  0.92549026,
         0.99607849,  0.41568631,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.32549021,  0.99215692,  0.81960791,  0.07058824,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.08627451,  0.91372555,
         1.        ,  0.32549021,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.50588238,  0.99607849,  0.9333334 ,  0.17254902,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.23137257,  0.97647065,
         0.99607849,  0.24313727,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.52156866,  0.99607849,  0.73333335,  0.01960784,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.03529412,  0.80392164,
         0.97254908,  0.227451  ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.49411768,  0.99607849,  0.71372551,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.29411766,  0.98431379,
         0.94117653,  0.22352943,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.07450981,  0.86666673,  0.99607849,  0.65098041,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.01176471,  0.7960785 ,  0.99607849,
         0.8588236 ,  0.13725491,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.14901961,  0.99607849,  0.99607849,  0.3019608 ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.12156864,  0.87843144,  0.99607849,
         0.45098042,  0.00392157,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.52156866,  0.99607849,  0.99607849,  0.20392159,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.2392157 ,  0.94901967,  0.99607849,
         0.99607849,  0.20392159,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.47450984,  0.99607849,  0.99607849,  0.8588236 ,  0.15686275,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.47450984,  0.99607849,
         0.81176478,  0.07058824,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ], dtype=np.float32)
real_data = real_data.reshape(1,28,28,1)

real_data2 = np.array([0,  1,  2,  3], dtype=np.float32)
real_data2 = real_data2.reshape(1,2,2,1)



real_filter = np.array([0,   1,   2,   3,   4,   5,   6,   7], dtype=np.float32)
real_filter = real_filter.reshape(2,2,1,2)


test = tf.nn.conv2d(real_data2, real_filter, strides=[1, 1, 1, 1], padding='SAME')

result = sess.run(test)



