import os
import multiprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import logging
import flwr as fl
logging.getLogger("flwr").setLevel(logging.CRITICAL)

from utils.   util_models import get_model_autoencoder, WeightsHashCallback

from pebble import concurrent
import ray

class FlClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test, dev_name, lab):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        print("Created FL client", dev_name, lab)
        #print(tf.config.list_physical_devices('GPU'))


    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        #print("Fit client")
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.0,
            verbose = 0, # default is 1
            #callbacks = [WeightsHashCallback()]
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            #"accuracy": history.history["accuracy"][0],
            #"val_loss": history.history["val_loss"][0], # TODO removed after put validation_split=0.0
            #"val_accuracy": history.history["val_accuracy"][0],
        }
    
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate GLOBAL model parameters on the local test data and return results

        #loss, accuracy = self.model.evaluate(self.x_test, self.x_test, 32, steps=steps)
        #num_examples_test = len(self.x_test)
        #return loss, num_examples_test, {"accuracy": accuracy}

        loss = self.model.evaluate(self.x_test, self.x_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test
    

  

#@concurrent.process(context=multiprocessing.get_context('spawn'))
@ray.remote
def start_flower_client(x, lab, dev_name = 'Dev', address = 'localhost', port = 8080):
    #tf.config.set_visible_devices([], 'GPU')

    x_train_portion = x
    #model = get_model_autoencoder(neurons=[100, 64, 32, 64,100]) # creato nel sottoprocesso  # custom 
    model = get_model_autoencoder() # creato nel sottoprocesso  

    fl_client = FlClient(model, x_train = x_train_portion, x_test = x_train_portion, dev_name=dev_name, lab = lab)

    print("Start flower client", dev_name, 'cluster', lab, 'port', port)

    # Create and start the Flower client
    fl.client.start_numpy_client(
        server_address=address + ":" + str(port),
        client=fl_client,
    )
    
    return model.get_weights() # needed if subprocess, use serialization
    #return model # error in serialization