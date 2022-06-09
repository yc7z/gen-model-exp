import torch
from experiment import Experiment


layers_lst = [
    [28*28, 32],
    [28*28, 256, 32]
]

weight_decay_lst = [
    1e-4,
    0
]

lr_lst = [
    0.05,
    0.01,
    0.005,
    0.001
]

batch_size_lst = [
    32, 
    64,
    128
]


for layers_idx in range(2): 
    for lr_idx in range(4): 
        for batch_size_idx in range(3):
            for weight_decay_idx in range(2):

                dataset_opts = {
                    'dataset': 'MNIST',
                    'dataset_path': '/home/jmackey/scratch/CJRepo/dataset' 
                }
                
                model_kwargs = {
                    'layers': layers_lst[layers_idx],
                    'bias': True,
                }
                
                model_opts = {
                    'model_name': "Autoencoder",
                    'model_type': 'AE',
                    'model_kwargs': model_kwargs
                }
                
                optim_kwargs = {
                    'weight_decay': weight_decay_lst[weight_decay_idx],
                }
                
                train_opts = { 
                    'optim': 'Adam', 
                    'lr': lr_lst[lr_idx],
                    'optim_kwargs': optim_kwargs,
                    'batch_size': batch_size_lst[batch_size_idx],
                    'epochs': 200, 
                    'save_every': 25,
                    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    'seed': 0,
                }

                results_opts = {
                    'training_results_path': './results',
                    'train_dump_file'   : 'training_results.json',
                    'model_weights_path': './model_weights',
                }

                opts = dict(dataset_opts, **model_opts)
                opts = dict(opts, **train_opts)
                opts = dict(opts, **results_opts)
                
                exp = Experiment(opts)
                exp.run()