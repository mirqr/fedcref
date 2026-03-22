
settings = {
    'seed': 42,
    'dataset_name': 'mnist',
    'num_clients': 10,
    'num_min_class': 2,
    'num_max_class': 4,
    'min_samples_per_class': 500,
    'max_samples_per_class': 600,
    'association_threshold': 0.25,
    'percentile_threshold': 75,
    'cluster_kind': 'oracle', # 'dirty_uniform', 'dirty_proximity', 'dec'
    'dirtiness_max': 0.5,
    'replace_sample': False,
    'start_port': 4000,
    'overlap' : 0,
    'experiment_group' : None, # only wandb
    'wandb_mode': 'online', # can be 'online', 'offline', or 'disabled'
    #'layerwise_pretrain_iters': 5000,
    #'overwrite': False,
    #'num_epochs': 10,
    #'batch_size': 32,
    #'lr': 0.001,
    #'momentum': 0.9,
    #'weight_decay': 0.0005
}

def update_settings(new_settings):
    settings.update(new_settings)