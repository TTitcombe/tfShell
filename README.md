# TF Shell

This project provides python classes to automate the building, training, and testing of tensorflow machine learning models.

The structure is as follows:
1. Create a class for the model architecture you would like to train. Example is in **Graphs/**
    - This class has an **__init__** method, which defines the weights, biases etc.
    - This class also has a **__call__** method, which takes as input some data, and return the output of the model
2. Create a *model* class which extends **Models/BaseModel**
    - In **__init__** of this class, the architecture class from (1.) is created and stored as an attribute
    - This class keeps track of the learning hyperparameters such as algorithm and learning rate. This is also where the tensor data placeholders are stored, as well as the ops to run during training and testing.
3. Creating a *train* class which extends **Training/BaseTrainer**
    - This class stores the logic for batching and training steps
    - (Not implemented yet) this class controls tensorboard
4. (Not implemented yet) a *test* class which provides unit tests for machine learning models

To see this structure in action, run **ann_example.py**. This loads a simple MLP and tries to learn the identity mapping (f(x) = x) - this is an experiment from the paper [Neural Arithmetic Logic Units](https://arxiv.org/pdf/1808.00508.pdf).

**The benefits of this structure**:
* You don't have to clog up your model code with training and testing logic
* To build and train a new model requires only a few lines to define weights and code the model structure. Nothing else needs to be touched
* Can edit training logic without having to touch the model


### TODO:
* Add a basic GAN model and training class
