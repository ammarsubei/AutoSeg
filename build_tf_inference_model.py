"""Builds the inference graph for the SQ-based model and saves the metagraph."""

import tensorflow as tf

def weight_variable(shape):
    """Returns a randomly-initialized weights tensor."""

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Returns a randomly-initialized bias tensor."""

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """Wrapper for tf.nn.conv2d."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """Wrapper for tf.nn.max_pool with kernel size 2x2 and stride 2."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def build_graph():
    """Builds the graph."""

    input_image = tf.placeholder(tf.float32, name='input_image')
    y_true = tf.placeholder(tf.float32, name='y_true')

    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([3, 3, 3, 64]) # kernel, kernel, input_channels, num_filters
        b_conv1 = bias_variable([64])

        h_conv1 = tf.nn.elu(conv2d(input_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("")
