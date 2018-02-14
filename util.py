import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def nameClient(index):
    name = u'simulator-{}'.format(index)
    return name.encode('utf-8')

def conv2d_layer(input, filter_size, out_dim, name, strides, func=lrelu):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.nn.conv2d(input, w, strides=[1,strides,strides,1], padding='VALID') + b
        if func is not None:
            output = func(output)

    return output

def dense_layer(input, out_dim, name, func=lrelu):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.matmul(input, w) + b
        if func is not None:
            output = func(output)

    return output

def maxpooling_layer(input, kern_size, strides, func=tf.nn.max_pool):
    return func(input, [1, kern_size, kern_size, 1], strides, padding="SAME")

def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])