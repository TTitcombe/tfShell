'''Test extrapolation failures of standard neural networks.
Replication of the experiment in section 1.1 of the paper
Neural Arithmetic Logic Units'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Training.base_trainer import VanillaTrainer
from Models.base_model import VanillaModel

def generate_data(n, min=-5., max=5., seed=42):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(min, max, size=(n,1))
    return x

def testing(x, x_val, hidden):
    results = {}
    for act_func in ['relu', 'tanh', 'leaky', 'sigmoid']:
        tf.reset_default_graph()
        print('Training {}...'.format(act_func))
        #define the model/graph first
        #use default hyperparams
        model = VanillaModel(1, 1, hidden, model_type=act_func)

        #pass the model into the trainer class
        trainer = VanillaTrainer(model)
        trainer.train(x, x, 100, 1000, x_val, x_val)
        #we are learning the identity function f(x) = x
        #so we are passing in x data as the target as well

        print("Testing...")
        final_output = trainer.get_output(x_val)[:,0]
        results[act_func] = abs(x_val[:,0] - final_output)

    for k, v in results.items():
        plt.scatter(x_val[:,0], v, label=k)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x = generate_data(1000) #uniform data in the range (-5,5)
    x_extrapolate = generate_data(1000, min=-20., max=20.)

    INPUT = 1
    OUTPUT = 1
    HIDDEN = [8, 8, 8]

    testing(x, x_extrapolate, HIDDEN)
