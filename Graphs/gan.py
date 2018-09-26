import tensorflow as tf
import numpy
from tensorflow.python.ops import variable_scope as vs

class D():
    def __init__(self, x_dim, hidden):
        self.hidden = hidden
        initialiser = tf.contrib.layers.xavier_initializer()
        old_h = x_dim
        with vs.variable_scope('D'):
            for j, h in enumerate(hidden):
                w_shape = (old_h, h)
                b_shape = (h,)
                tf.get_variable('W{}'.format(j), w_shape, initializer=initialiser)
                tf.get_variable('b{}'.format(j), dtype=tf.float32, initializer=tf.zeros(b_shape))
                old_h = h
            w_shape = (old_h, 1)
            b_shape = (1,)
            tf.get_variable('Wout', w_shape, initializer=initialiser)
            tf.get_variable('bout', dtype=tf.float32, initializer=tf.zeros(b_shape))


    def __call__(self, input, alpha = 0.2):
        with vs.variable_scope('D', reuse=True):
            for j, h in enumerate(self.hidden):
                w = tf.get_variable('W{}'.format(j))
                b = tf.get_variable('b{}'.format(j))
                net = tf.matmul(input, w) + b
                input = tf.nn.leaky_relu(net, alpha = alpha)

            w = tf.get_variable('Wout')
            b = tf.get_variable('bout')
            net = tf.matmul(input, w) + b
            act = tf.nn.sigmoid(net, name='out')

        return net, act


class G():
    def __init__(self, z_dim, x_dim, hidden):
        self.hidden = hidden
        initialiser = tf.contrib.layers.xavier_initializer()
        old_h = z_dim
        with vs.variable_scope('G'):
            for j, h in enumerate(hidden):
                w_shape = (old_h, h)
                b_shape = (h,)
                tf.get_variable('W{}'.format(j), w_shape, initializer=initialiser)
                tf.get_variable('b{}'.format(j), dtype=tf.float32, initializer=tf.zeros(b_shape))
                old_h = h
            w_shape = (old_h, x_dim)
            b_shape = (x_dim,)
            tf.get_variable('Wout', w_shape, initializer=initialiser)
            tf.get_variable('bout', dtype=tf.float32, initializer=tf.zeros(b_shape))

    def __call__(self, input, alpha = 0.2):
        with vs.variable_scope('G', reuse=True):
            for j, h in enumerate(self.hidden):
                w = tf.get_variable('W{}'.format(j))
                b = tf.get_variable('b{}'.format(j))
                net = tf.matmul(input, w) + b
                input = tf.nn.leaky_relu(net, alpha = alpha)

            w = tf.get_variable('Wout')
            b = tf.get_variable('bout')
            net = tf.matmul(input, w) + b
            act = tf.nn.tanh(net, name='out')

        return net, act
