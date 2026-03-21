import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


from util_models import get_model_autoencoder 



import time
import fl_server
import fl_client

m = get_model_autoencoder()
del m

fl_server.my_create_server_subproc("localhost", port=str(5000+1), num_clients=2, num_rounds=10)
time.sleep(5)
fut1 = fl_client.start_flower_client(None,1)
fut2 = fl_client.start_flower_client(None,2)

fut1.result()
fut2.result()
