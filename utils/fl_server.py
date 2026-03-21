import os
import multiprocessing
from pebble import concurrent
import ray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf


import logging
import flwr as fl

logging.getLogger("flwr").setLevel(logging.CRITICAL)


from utils.util_models import get_model_autoencoder
#import run_config




def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 2 if server_round < 2 else 7, # was else 2
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}



# MAIN PART.
# server is not a class, but a function. 
def main(address: str, port:str, num_clients: int, num_rounds: int) -> None: # 
    #print("Starting server, expecting", num_clients, "clients")

    
    #model = get_model_autoencoder(neurons=[100, 64, 32, 64,100]) # custom
    model = get_model_autoencoder()
    print("Server model created")

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
            fraction_fit=1, #0.3,
            min_available_clients=num_clients,
            min_fit_clients=num_clients,
            
            fraction_evaluate=0.0, # disabled
            min_evaluate_clients=2,

            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
            #evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config
    )

    print('address', address, 'port', port)
    fl.server.start_server(
        server_address=address+":"+port,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return None


def my_create_server(address: str, port:str, num_clients: int, num_rounds: int):
    print("Starting server calling subproc, expecting", num_clients, "clients")
    main(address, port, num_clients, num_rounds)

#@concurrent.process(context=multiprocessing.get_context('spawn'))
@ray.remote
def my_create_server_subproc(address: str, port:str, num_clients: int, num_rounds: int):
    #tf.config.set_visible_devices([], 'GPU')
    #print("Starting server, expecting", num_clients, "clients", "port", port)
    print("Starting server address ", address, "port ", port, "expecting ", num_clients, "clients")
    main(address=address, port=port, num_clients=num_clients, num_rounds=num_rounds)
    # hide GPU from tf
    




#if __name__ == "__main__":
    #main(run_config.address, run_config.port, run_config.num_clients, run_config.num_rounds)

