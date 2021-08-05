import tensorflow as tf
import numpy as np
import matplotlib as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

fig, ax = plt.subplots(10, 10)

k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(mnist.train.images[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()

print(f'Shape of feature matrix: {mnist.train.images.shape}')
print(f'Shape of target matrix: {mnist.train.labels.shape}')

print(f'One-hot encoding for 1st observation: {mnist.tran.labels[0]}')

x_train = tf.placeholder('float', [None, 784])
W = tf.Variables(tf.zeros([784, 10]))
b = tf.Variables(tf.zeros([10]))

y = tf.n.softmax(tf.matmul(x_train, W) + b)
y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_train: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={x_train: mnist.test.images, y_: mnist.test.labels}))
