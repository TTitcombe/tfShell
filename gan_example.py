import numpy as np
import tensorflow as tf

from Models.gan_model import GANModel
from Training.gan_trainer import GANTrainer

IMAGE_PATH = './celeba/' #DIRECTORY TO IMAGES TO TRAIN ON. 
BATCH_SIZE = 256
N_EPOCHS = 100
PRINT_EVERY = 1

x_dim = 178 * 218 * 3 #because I'm using celeba
z_dim = 100

model = GANModel(x_dim, z_dim)
trainer = GANTrainer(model)

print("Training a GAN..")
trainer.train(IMAGE_PATH, BATCH_SIZE, N_EPOCHS, PRINT_EVERY)
