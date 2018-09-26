import tensorflow as tf
import os
import numpy as np

from Graphs.gan import D, G
from Models.base_model import BaseModel


class GANModel(BaseModel):
    '''Class to store the learning parameters and ops for a basic GAN'''

    def __init__(self,x_dim, z_dim, hidden_D = [500], hidden_G = [500], hyperparams = {}):
        '''
        x_dim | int, dimension of input data
        z_dim | int, dimension of noise data (G input)
        hidden_D | list of ints, size of hidden layers
        hidden_G | list of ints, size of hidden layers of G network
        hyperparams | dict, keys 'G' and 'D'. Each key has a value of
                    a dict. hyperparams['G'] contains G network hyperparams
        '''
        self.z_dim = z_dim
        self.__set_hypers(hyperparams)
        self.__placeholders(x_dim, z_dim)
        self.D = D(x_dim, hidden_D)
        self.G = G(z_dim, x_dim, hidden_G)
        self.__ops()

    def __set_hypers(self, hyperparams):
        hyperG = hyperparams.get('G', {})
        hyperD = hyperparams.get('D', {})

        self.G_global_step = tf.Variable(0, trainable=False)
        self.D_global_step = tf.Variable(0, trainable=False)

        G_optim = hyperG.get('optim', 'gd')
        D_optim = hyperD.get('optim', 'gd')

        G_decay = hyperG.get('decay', None)
        D_decay = hyperD.get('decay', None)
        if G_decay is not None:
            start_lr = hyperG.get('lr', 0.001)
            self.G_lr = tf.train.exponential_decay(start_lr, self.G_global_step,
                                                       decay_steps=10, decay_rate=G_decay, staircase=True)
        else:
            self.G_lr = hyperG.get('lr', 0.001)
        if D_decay is not None:
            start_lr = hyperD.get('lr', 0.001)
            self.D_lr = tf.train.exponential_decay(start_lr, self.D_global_step,
                                                        decay_steps=10, decay_rate = D_decay, staircase=True)
        else:
            self.D_lr = hyperD.get('lr', 0.001)

        if G_optim.lower() == 'adam':
            self.G_optim = tf.train.AdamOptimizer(self.G_lr)
        elif G_optim.lower() == 'gd':
            self.G_optim = tf.train.GradientDescentOptimizer(self.G_lr)
        else:
            raise NotImplementedError("Only Adam and standard GD at the moment")

        if D_optim.lower() == 'adam':
            self.D_optim = tf.train.AdamOptimizer(self.D_lr)
        elif D_optim.lower() == 'gd':
            self.D_optim = tf.train.GradientDescentOptimizer(self.D_lr)

    def __placeholders(self, x_dim, z_dim):
        self.x = tf.placeholder(tf.float32, [None, x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, z_dim], name='z')

    def __ops(self):
        self.fake_logits, self.fake_image = self.G(self.z)
        self.D_logits, self.D_act  = self.D(self.x)
        self.D_logits_fake, self.D_act_fake = self.D(self.fake_image)

        #loss
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D_logits)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.zeros_like(self.D_logits_fake)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.ones_like(self.D_logits_fake)))

        #Optimisers
        train_vars = tf.trainable_variables()

        d_vars = [var for var in train_vars if 'D' in var.name]
        g_vars = [var for var in train_vars if 'G' in var.name]

        self.optimD = self.D_optim.minimize(self.D_loss, var_list=d_vars)
        self.optimG = self.G_optim.minimize(self.G_loss, var_list=g_vars)

    def sampleZ(self, n):
        '''
        Generates the noisy input for the Generator
        n | int, number of samples to draw
        '''
        shape = (n, self.z_dim)
        z = np.random.normal(0.0, 0.5, size=shape)
        z[z > 1] = 1.
        z[z < -1] = -1.
        return z
