import numpy as np
import tensorflow as tf

class MLP():
    '''
    A MLP in which the hidden layers can have one of several activation functions,
    but the output layer does not have an activation function.
    This model was created to replicate the identity function extrapolation failure
    experiment, as seen in the paper 'Neural Arithmetic Logic Units' by Trask et al.
    '''
    ACT_FUNCS = {'relu': tf.nn.relu,
                    'relu6': tf.nn.relu6,
                    'elu': tf.nn.elu,
                    'leaky': tf.nn.leaky_relu,
                    'sigmoid': tf.sigmoid,
                    'tanh': tf.tanh,
                    'softplus': tf.nn.softplus,
                    'None': None}

    def __init__(self, input_dim, output_dim, hidden_dim = [], act_func='relu'):
        self.__n_layers = len(hidden_dim)
        self.act_func = MLP.ACT_FUNCS[act_func]

        #initialiser = tf.contrib.layers.xavier_initializer()
        initialiser = tf.truncated_normal_initializer()
        with tf.variable_scope('MLP'):
            for i, dim in enumerate(hidden_dim):
                if i == 0:
                    inp_dim = input_dim
                else:
                    inp_dim = hidden_dim[i-1]
                W = tf.get_variable("W{}".format(i), [inp_dim, dim], initializer=initialiser)
                b = tf.get_variable('b{}'.format(i), [dim,], initializer=tf.zeros_initializer())
            W = tf.get_variable("Wout", [hidden_dim[-1], output_dim], initializer=initialiser)
            b = tf.get_variable('bout', [output_dim,], initializer=tf.zeros_initializer())

    def __call__(self, x_input):
        with tf.variable_scope('MLP', reuse=True):
            for i in range(self.__n_layers):
                W = tf.get_variable("W{}".format(i))
                b = tf.get_variable("b{}".format(i))
                x_input = tf.add(tf.matmul(x_input, W), b)
                if self.act_func is not None:
                    x_input = self.act_func(x_input)

            W = tf.get_variable("Wout")
            b = tf.get_variable("bout")
            x_input = tf.add(tf.matmul(x_input, W), b)

        return x_input
