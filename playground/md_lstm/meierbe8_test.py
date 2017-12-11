""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
This example is using TensorFlow layers API, see 'convolutional_network_raw'
example for a raw implementation with variables.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

from playground.md_lstm import PyramidCell2D
from scipy.misc import imresize

use_mnist = False

if use_mnist:

    # Training Parameters
    learning_rate = 0.001
    num_steps = 2000
    batch_size = 16

    # Network Parameters
    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

else:

    # Training Parameters
    learning_rate = 0.001
    num_steps = 2000
    batch_size = 1

    # Network Parameters
    num_input = 128 * 128  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

import numpy as np

def process_dimension(input, cell, dim, scope):
    """
    Processes the image in a given dimension
    :param input:
    :param cell:
    :param dim:
    :param scope:
    :return:
    """
    act_img = input
    # flip dimension
    if np.sign(dim) < 0:
        act_img = tf.reverse(act_img, [np.abs(dim)])

    # transpose to make the relevant dim, dim1
    if np.abs(dim) > 1:
        perm = list(range(len(act_img.shape)))
        perm[1] = np.abs(dim)
        perm[np.abs(dim)] = 1
        act_img = tf.transpose(act_img, perm)


    hidden  = cell.zero_state(batch_size, tf.float32)
    outputs = []
    # use tf.loop here
    for i in range(input.shape.as_list()[1]):
        out, hidden = cell(act_img[:, i], hidden, dim, scope)
        outputs.append(out)


    print("process dimension")
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2, 3])
    return outputs

def allocate_pyramid_cell(dims, kernel_size, state_size, dense_hidden, input, scope_name):
    """
    Allocates on pyramid cell and processes the inputs in all dimensions according to dims
    :param dims:
    :param kernel_size:
    :param state_size:
    :param dense_hidden:
    :param input:
    :param scope_name:
    :return:
    """

    # allocate pyramid cell
    with tf.variable_scope(scope_name, initializer=tf.random_uniform_initializer(-.01, 0.1)) as scope:
        cell = PyramidCell2D.BasicPyramidLSTMCell2D(input.get_shape().as_list()[1:3], kernel_size, state_size)
        cell.init_variables(dims, input[:, 1].get_shape().as_list()[2], scope)

    # for all dimensions
    processed_dims = []
    # process dims
    print("D:{}".format(dims))
    for dim in dims:
        output = process_dimension(input, cell, dim, scope)
        processed_dims.append(output)

    print("P:{}".format(processed_dims))

    processed_dims = tf.add_n(processed_dims)

    # return processed_dims
    #
    # fully-connected
    out_dense = tf.layers.dense(inputs=processed_dims, units=dense_hidden, activation=tf.nn.tanh)

    return out_dense

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]

        if use_mnist:
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
        else:
            x = tf.reshape(x, shape=[-1, 128, 128, 1])

        # # Convolution Layer with 32 filters and a kernel size of 5
        # conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv1 = x

        for i in range(1):
            state_size = 64
            dims = np.array(range(1, len(conv1.get_shape().as_list()) - 1))
            dims = np.concatenate((dims, dims * -1))
            conv1 = allocate_pyramid_cell(dims, [5], 4, state_size, conv1, "pyramid_{}".format(i))

        conv2 = conv1

        # # Convolution Layer with 64 filters and a kernel size of 3
        # conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

if use_mnist:
    train_imgs = mnist.train.images
else:
    print(mnist.train.images.dtype)
    print(np.min(mnist.train.images))
    print(np.max(mnist.train.images))
    train_imgs = np.reshape(np.asarray(list(map(lambda x: imresize(x * 255, (128, 128)), np.reshape(mnist.train.images, (-1, 28, 28))))), (-1, 128 * 128)).astype(np.float32) / 255
    print(train_imgs.dtype)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_imgs}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
