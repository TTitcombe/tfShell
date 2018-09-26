import tensorflow as tf
import numpy as np
import os


class BaseTrainer():
    '''
    A class to handle the batching and training of a defined model.
    This is the default model. Extend this class and complete the remaining
    methods.
    '''

    def __init__(self, model):
        '''
        model | A fully defined model from the Models directory.
                A class in which the placeholders are stored.
        '''
        self._Sess = None
        self.model = model

        if self._Sess is None:
            self._Sess = tf.Session()
            init = tf.global_variables_initializer()
            self._Sess.run(init)

    def train(self, batchSize, print_every=100):
        '''
        The main training loop.
        '''
        print_every = max(0, print_every)
        assert int(print_every) == print_every, "Epoch number must be int"

        self.bs = batchSize

    def __train_epoch(self):
        raise NotImplementedError("Must define the training loop for an epoch")

    def __train_step(self):
        raise NotImplementedError("Must define the training step")

    def validate(self):
        '''
        Define and run the ops to produce output from the network
        without training it
        '''
        raise NotImplementedError("Need to define validation")

    def get_output(self):
        '''
        Method to produce output from the network for post processing,
        without calculating loss, error etc. or training the network.
        '''
        raise NotImplementedError("Need to define method for producing output")



class VanillaTrainer(BaseTrainer):
    '''
    A training class for a generic machine learning model
    e.g. a simple feedforward neural network, which takes x as input and
    produces y as output. Not suitable for more exotic models e.g. GANs.
    '''

    def train(self, x, y, batchSize, n_epochs, x_val = None, y_val=None,
                print_every=100):
        '''
        Inputs:
            x | np array of input data
            y | np array of target data
            batchSize | int
            n_epochs | int, the number of epochs to train for
            x_val | None or a np array of input validation data
            y_val | None or a np array of target validation data
        '''
        super().train(batchSize, print_every)

        self.n_steps = x.shape[0] // batchSize
        if x.shape[0] / batchSize != float(self.n_steps):
            self.n_steps += 1

        for epoch in range(n_epochs):
            tr_err, tr_loss = self.__train_epoch(epoch, x, y)
            if epoch % print_every == 0:
                print("Epoch: {}".format(epoch))
                print("Training err: {}".format(tr_err))
                if x_val is not None:
                    te_err, te_loss = self.validate(x_val, y_val)
                    print("Testing err: {}".format(te_err))
                print("\n")

    def __train_epoch(self, epoch, x, y):
        '''
        The epoch training loop. there are K steps, dependent on batch
        size and number of data points. For each of the K steps, run a training
        step. Return the most recent training step for printing
        '''
        for step in range(self.n_steps):
            step_start = self.bs * step
            step_end = self.bs * (step + 1)
            x_batch = x[step_start:step_end]
            y_batch = y[step_start:step_end]

            i = epoch * self.n_steps + step
            train_error, train_loss = self.__train_step(i, x_batch, y_batch)
        return train_error, train_loss

    def __train_step(self, i, x, y):
        '''
        A single training step.
        i | int, the current step
        x | batch of input data
        y | batch of target data
        '''
        feed_dict = {self.model.x: x,
                    self.model.y: y}
        to_run = [self.model.error, self.model.loss, self.model.optimise]
        _err, _loss, _ = self._Sess.run(to_run, feed_dict)
        return _err, _loss

    def validate(self, x, y):
        '''
        Calculate error and loss without training
        '''
        feed_dict = {self.model.x: x,
                    self.model.y: y}
        to_run = [self.model.error, self.model.loss]
        _err, _loss = self._Sess.run(to_run, feed_dict)
        return _err, _loss

    def get_output(self, x):
        feed_dict = {self.model.x: x}
        y_hat = self._Sess.run(self.model.y_hat, feed_dict)
        return y_hat
