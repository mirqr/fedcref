import os
import json
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

BATCH_SIZE = 150  # number of changes before saving to file
SAVE_INTERVAL = 30  # seconds before saving to file

cache = {}
pending_changes = {}
last_save_time = time.time()

from filelock import FileLock

def ensure_cache_directory_exists():
    if not os.path.exists("cache"):
        os.makedirs("cache")

def get_device_from_key(key):
    return key.split('-')[0]

def get_cache_file_for_device(device):
    return f'cache/inferences_cache_{device}.json'

def load_cache(device):
    ensure_cache_directory_exists()
    
    data = None
    lock = FileLock(get_cache_file_for_device(device) + ".lock")
    with lock:
        if os.path.exists(get_cache_file_for_device(device)):
            with open(get_cache_file_for_device(device), 'r') as f:
                data = f.read()
    if data:
        cache[device] = json.loads(data)
    else:
        cache[device] = {}

def save_cache_to_file(device):
    ensure_cache_directory_exists()

    lock = FileLock(get_cache_file_for_device(device) + ".lock")
    with lock:
        with open(get_cache_file_for_device(device), 'w') as f:
            json.dump(cache[device], f)

def get_inference_from_cache(key):
    device = get_device_from_key(key)
    if device not in cache:
        load_cache(device)
    return cache[device].get(key)


def append_to_cache(data_dict):
    """
    Append a dictionary of key-values to the appropriate cache file for the device.

    :param data_dict: Dictionary containing key-value pairs to be appended to the cache.
    """
    # Assuming all keys in the dictionary are from the same device,
    # we'll get the device from the first key.
    device = get_device_from_key(next(iter(data_dict.keys())))

    # Load the current cache for the device if not already loaded
    if device not in cache:
        load_cache(device)

    # Update the cache with the new data
    cache[device].update(data_dict)

    # Save the updated cache to file
    print(f'Saving cache for {device} to file')
    save_cache_to_file(device)

def save_inference_to_cache(key, value, BATCH_SIZE=BATCH_SIZE):
    global last_save_time
    device = get_device_from_key(key)

    if device not in cache:
        load_cache(device)

    if device not in pending_changes:
        pending_changes[device] = 0

    cache[device][key] = value
    pending_changes[device] += 1

    # Check if we should save based on BATCH_SIZE or SAVE_INTERVAL
    if pending_changes[device] >= BATCH_SIZE or (time.time() - last_save_time) >= SAVE_INTERVAL:
        print(f'Saving cache for {device} to file')
        save_cache_to_file(device)
        pending_changes[device] = 0
        last_save_time = time.time()




import hashlib
model_hash_cache = {}

def get_model_hash_with_cache(model):
    # Assuming the model has an 'id' or some unique identifier
    model_id = id(model)
    
    if model_id in model_hash_cache:
        return model_hash_cache[model_id]
    #print('Model not in cache, computing hash')
    model_hash_value = get_model_hash(model)  # Original hash function
    model_hash_cache[model_id] = model_hash_value
    return model_hash_value

def get_model_hash(model):
    # Convert model architecture to string
    #model_arch_str = str(model.get_config())

    # Convert model weights to bytes
    model_weights = [w.tobytes() for w in model.get_weights()]
    model_weights_bytes = b''.join(model_weights)

    # Concatenate architecture and weights and compute SHA256 hash
    #combined_data = model_arch_str.encode('utf-8') + model_weights_bytes
    combined_data = model_weights_bytes
    return hashlib.sha256(combined_data).hexdigest()

def get_model_autoencoder(neurons=[100,64, 32, 64,100]):    
    #m = AutoencoderFlat(n_features=784, hidden_neurons=[16, 8, 16])
    m = AutoencoderFlat(n_features=784, dropout_rate=0, hidden_neurons=neurons) # THIS 
    #m = AutoencoderConv_new(n_features=784) 
    #m = AutoencoderFlat(n_features=784, hidden_neurons=[64, 32, 16, 32, 64])
    #m = AutoencoderConvV2() 
    m.build(input_shape=(None,784))
    m.compile(optimizer='adam', loss='mse')
    #print('Model compiled')
    # print model summary
    #print(m.summary())

    return m

class WeightsHashCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        # Compute the weights hash when training ends
        self.model.weights_hash = self.compute_weights_hash() # callback have access to model via self.model 


    def compute_weights_hash(self):
        # Retrieve the current weights as a single numpy array
        weights = self.model.get_weights()
        weights_np = np.concatenate([w.flatten() for w in weights])

        # Compute the hash of the current weights
        current_hash = hashlib.sha256(weights_np.tobytes()).hexdigest()

        return current_hash

class AutoencoderFlat(tf.keras.Model): 
    def __init__(self, n_features, dropout_rate=0.1, hidden_neurons=[64, 32, 64]):
        super(AutoencoderFlat, self).__init__()
        self.model_name = 'Mod'
        self.weights_hash = None # TODO
        self.hashh = None

        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate

        layers_list = []

        # If dropout is desired at the input
        if self.dropout_rate > 0:
            layers_list.append(layers.Dropout(self.dropout_rate))

        for neu in self.hidden_neurons:
            layers_list.append(layers.Dense(neu, activation='relu'))
            if self.dropout_rate > 0:
                layers_list.append(layers.Dropout(self.dropout_rate))

        layers_list.append(layers.Dense(n_features, activation='sigmoid'))

        self.model_layers = tf.keras.Sequential(layers_list)

    def call(self, x):
        return self.model_layers(x)
    
    def set_model_name(self, name):
        self.model_name = name

    def get_model_signature(self):
        '''Returns a string describing the model's architecture and characteristics'''
        hidden_layers_signature = "-".join(map(str, self.hidden_neurons))
        return f"AEFlat_Features-{self.n_features}_Dropout-{self.dropout_rate}_Neurons-{hidden_layers_signature}"
    
    def get_model_hash(self):
        if self.hashh is None:
            signature = self.get_model_signature()
            model_weights = [w.tobytes() for w in self.get_weights()] # list of 
            model_weights_bytes = b''.join(model_weights)
            combined_data = signature.encode('utf-8') + model_weights_bytes
            hash = hashlib.sha256(combined_data).hexdigest()
            self.hashh = hash
        #self.hashh = self.weights_hash # TODO not necessary anymore, but keep it for now to avoid changing the code
        return self.hashh

class AutoencoderConv_new(tf.keras.Model): 
    def __init__(self, n_features=784, latent_dim=32, dropout_rate=0.1):
        super(AutoencoderConv_new, self).__init__()
        self.model_name = 'ConvAEFlat'
        self.weights_hash = None
        self.hashh = None

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Encoder
        encoder_inputs = layers.Input(shape=(n_features,))
        x = layers.Reshape((28, 28, 1))(encoder_inputs)
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, activation="relu")(x)
        self.encoder = tf.keras.Model(encoder_inputs, latent, name="encoder")

        # Decoder
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(7*7*64, activation="relu")(decoder_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        outputs = layers.Flatten()(x)  # back to 784
        self.decoder = tf.keras.Model(decoder_inputs, outputs, name="decoder")

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def set_model_name(self, name):
        self.model_name = name

    def get_model_signature(self):
        return f"ConvAEFlat_Features-{self.n_features}_Latent-{self.latent_dim}_Dropout-{self.dropout_rate}"
    
    def get_model_hash(self):
        if self.hashh is None:
            signature = self.get_model_signature()
            model_weights = [w.tobytes() for w in self.get_weights()]
            model_weights_bytes = b''.join(model_weights)
            combined_data = signature.encode('utf-8') + model_weights_bytes
            hash = hashlib.sha256(combined_data).hexdigest()
            self.hashh = hash
        return self.hashh



