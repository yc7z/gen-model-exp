import sys

import random
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from os import makedirs
from utils.dump import DumpJSON
from utils.meters import AccuracyMeter, MSEMeter, AverageMeter

from vae.elbo import ELBO
from torch.nn.functional import mse_loss

import torch.optim as optim

from autoencoder.autoencoder import Autoencoder, AutoencoderCNN
from vae.vae import VariationalAutoencoder, VariationalAutoencoderCNN


class Experiment:
    def __init__(self, opts):
        for key, value in opts.items():
            setattr(self, key, value)
    
        try:
            makedirs(self.training_results_path)
            makedirs(self.model_weights_path)
        except:
            pass
        
        # model config string
        self.configuration = f"model={self.model_name},"
        
        for key in self.model_kwargs:
            self.configuration += f"{key}={self.model_kwargs[key]}".replace(",", "-").replace("[", "").replace("]", "") + ","
            
        self.configuration += f"batch={self.batch_size},lr={self.lr},wd={self.optim_kwargs['weight_decay']}"
        
        # datasets and loaders
        dataset = getattr(datasets, self.dataset)
        
        train_data = dataset(root=self.dataset_path, 
                             train=True, 
                             download=False,
                             transform=T.ToTensor())
        test_data = dataset(root=self.dataset_path, 
                            train=False, 
                            download=False,
                            transform=T.ToTensor())
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        
        # model
        self.model = globals()[self.model_name](**self.model_kwargs)
        self.model.to(self.device)
        
        # optimizer and lr scheduler
        self.optimizer = getattr(optim, self.optim)(params=self.model.parameters(), 
                                                    lr=self.lr,
                                                    **self.optim_kwargs)
        
        self.lr_scheduler = None
            

    def run(self):
        # seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # meters
        meters = {
            'loss': AverageMeter(),
            'mse': MSEMeter(root=True),
        }

        best_test_loss = 1e10
        model_path = self.configuration
        
        # starts at the last epoch
        for epoch in range(1, self.epochs + 1):
            
            # json dump file
            results_path = self.training_results_path + '/' + self.train_dump_file
            results = DumpJSON(read_path=results_path, write_path=results_path)
            
            for name in meters:
                meters[name].reset()
            
            # train
            results = self.run_epoch("train",
                                     epoch,
                                     self.train_loader,
                                     results, 
                                     meters)  
            
            for name in meters:
                meters[name].reset()
                
            # test
            results = self.run_epoch("test",
                                       epoch,
                                       self.test_loader,
                                       results, 
                                       meters)
            
            # if model gets new best test value, save it
            if meters['loss'].value() <= best_test_loss:
                best_test_loss = meters['loss'].value()
            
                torch.save({'epoch': epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'test_loss': meters['loss'].value()},
                           self.model_weights_path + "/best_" + model_path + ".pt")
            
            # save every X model weights
            if epoch == 1 or epoch % self.save_every == 0:
                torch.save({'epoch': epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'test_loss': meters['loss'].value()},
                            self.model_weights_path + f"/epoch{epoch}_" + model_path + ".pt")
            
            # dump to json
            results.save()
        
        results.to_csv()
            
    def run_epoch(self,
                  phase,
                  epoch,
                  loader,
                  results, 
                  meters):
        
        # switch phase
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise Exception('Phase must be train, test or analysis!')    
        
        for i, (X, y) in enumerate(loader, 1):
            # input images
            X = X.to(self.device)
            
            if self.model_type == "AE":
                # run model on input and evaluate loss
                X_hat = self.model(X)
                loss = mse_loss(input=X, target=X_hat, reduction='sum')
                
                # compute gradient and do optimization step
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # adjust learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                # record statistics
                meters['loss'].add(float(loss.item()))
                meters['mse'].add(X, X_hat, n=X.shape[0])
            
            elif self.model_type == "VAE":
                # run model on input and evaluate loss
                X_hat, mu, logvar = self.model(X)
                loss = ELBO(x=X, x_hat=X_hat, mu=mu, logvar=logvar, gamma=self.gamma, reduction='sum')
                
                # compute gradient and do optimization step
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                  
                # adjust learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                # record statistics
                meters['loss'].add(float(loss.item()))
            
            elif self.model_type == "GAN":
                pass
            
            # append row to results CSV file
            if results is not None:
                if i == len(loader):
                    
                    stats = {'phase': phase,
                             'epoch': epoch,
                             'iters': len(loader),
                             'iter_loss': meters['loss'].val,
                             'avg_loss': meters['loss'].avg,
                             'rmse': meters['mse'].value()
                             }
                    
                    stats = dict(stats, **self.model_kwargs)
                    stats = dict(stats, **self.optim_kwargs)

                    results.append(dict(self.__getstate__(), **stats))
            
        output = '{}\tLoss: {meter.val:.4f} ({meter.avg:.4f})\tEpoch: [{}/{}]\t'.format(phase.capitalize(), epoch, self.epochs, meter=meters['loss'])

        print(output)
        sys.stdout.flush()
                    
        return results
    
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = [
                      'train_loader',
                      'test_loader',
                      'model',
                      'loss',
                      'optimizer',
                      'lr_scheduler',
                      'dataset_path',
                      'device',
                      'seed',
                      'model_kwargs',
                      'optim_kwargs'
                      ]
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state
    
