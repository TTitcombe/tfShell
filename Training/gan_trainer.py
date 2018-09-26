import tensorflow as tf
import numpy
from tensorflow.python.ops import variable_scope as vs
import os

from Training.base_trainer import BaseTrainer
from Utils.image_loader import ImageLoader

class GANTrainer(BaseTrainer):
    '''
    A basic trainer class for Generative Adversarial Networks.
    GANs differ from a generic neural network in that there are two
    models, a descriminator (D) and generator (G). The training step
    in a GAN requires evaluation of both the D and G nets.
    '''

    def train(self, image_path, batchSize, n_epochs, print_every=100):
        '''
        Inputs:
            image_path | str, path to directory where images to train are stored
            batchSize | int
            n_epochs | int, the number of epochs to train for
            print_every | int, number of epochs inbetween printing training status
        '''
        super().train(batchSize, print_every)
        self.image_loader = ImageLoader(image_path, batchSize)
        self.n_steps = self.image_loader.steps

        for epoch in range(n_epochs):
            Dloss, Gloss = self.__train_epoch(epoch)
            if epoch % print_every == 0:
                print("Epoch: {}".format(epoch))
                print("D loss: {}; G loss: {} \n".format(Dloss, Gloss))

    def __train_epoch(self, epoch):
        for step in range(self.n_steps):
            i = epoch * self.n_steps + step
            Dloss, Gloss = self.__train_step(i)
        return Dloss, Gloss

    def __train_step(self, i):
        x = self.image_loader.next_batch()
        z = self.model.sampleZ(self.bs)

        feed_dict = {self.model.x: x,
                    self.model.z: z}

        Dloss, Gloss, _, _ = self._Sess.run(
                            [self.model.D_loss, self.model.G_loss,
                                self.model.optimD, self.model.optimG],
                                 feed_dict)
        return Dloss, Gloss

    def get_output(self, n):
        z = self.sampleZ(n)
        feed_dict = {self.model.z: z}
        y_hat = self._Sess.run(self.model.fake_image, feed_dict)
        return y_hat
