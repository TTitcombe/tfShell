import numpy as np
import tensorflow as tf

from Graphs.mlp import MLP

class BaseModel():
    '''
    A class to store the learning hyperparameters of the network
    e.g. learning rate, learning algorithm, decay rate etc.
    '''
    def __init__(self, hyperparams = {}):
        '''
        Inputs:
            model | a class from the Graph directory. A tensorflow model e.g.
                CNN, with __init__ and __call__ methods only.
            intput_dim | int, input dimension of the model
            output_dim | int
            hyperparams | dict
        '''
        self.__set_hypers(hyperparams)

    def __placeholders(self):
        raise NotImplementedError("Define your tensor placeholders here")

    def __set_hypers(self, hyper):
        '''
        Set any hyperparameters of the overall model. E.g. learning rate
        and algorithm.
        Default is:
            gradient descent, 0.001 lr, no decay.
        '''
        self.global_step = tf.Variable(0, trainable=False)
        optim = hyper.get('optim', 'gd')
        decay = hyper.get('decay', None)
        start_lr = hyper.get('lr', 1e-3)

        if decay is not None:
            self.lr = tf.train.exponential_decay(start_lr, self.global_step,
                                                       decay_steps=1000, decay_rate=decay, staircase=True)
        else:
            self.lr = start_lr

        if optim.lower() == 'adam':
            self.optim = tf.train.AdamOptimizer(self.lr)
        elif optim.lower() == 'gd':
            self.optim = tf.train.GradientDescentOptimizer(self.lr)
        elif optim.lower() == 'rms':
            self.optim = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise NotImplementedError("Learning algorithm not recognised")

    def __ops(self):
        raise NotImplementedError("Define the ops you want to call during training and testing")


class VanillaModel(BaseModel):
    '''
    The model class for a generic feedforward NN, minimising the mse, and
    is monitoring  mean absolute error as an error metric.
    In a classification task this would be e.g. cross entropy and accuracy
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, model_type, hyperparams = {}):
        super().__init__(hyperparams)

        self.__placeholders(input_dim, output_dim)
        self.model = MLP(input_dim, output_dim, hidden_dim, model_type)
        self.__ops()

    def __placeholders(self, input_dim, output_dim):
        self.x = tf.placeholder(tf.float32, [None, input_dim], name='input')
        self.y = tf.placeholder(tf.float32, [None, output_dim], name='ouput')

    def __ops(self):
        self.y_hat = self.model(self.x)
        self.error = tf.reduce_mean(tf.abs(self.y_hat - self.x), name='mean_abs_error')
        self.square = tf.square(tf.subtract(self.y_hat, self.y), name='square_diffs')
        self.loss = tf.reduce_mean(self.square, name='loss')
        self.optimise = self.optim.minimize(self.loss)
