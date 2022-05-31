import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class VariationalAutoencoderCNN(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoderCNN, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1), # shape (B, 64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # shape (B, 128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # shape (B, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=latent_dim, kernel_size=4, stride=1, padding=0) # shape (B, latent_dim, 1, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=4, stride=1, padding=0, output_padding=0), # shape (B, 256, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=0), # shape (B, 128, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), # shape (B, 64, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), # shape (B, 1, 28, 28)
            nn.Sigmoid()
        )
        
        self.linear_mu = nn.Linear(in_features=64, out_features=latent_dim)
        self.linear_logvar = nn.Linear(in_features=64, out_features=latent_dim)
        
        self.linear_dec = nn.Linear(in_features=latent_dim, out_features=64)
        
    def forward(self, x):
        """
        x.shape = (B, 1, 28, 28)
        """
        
        # Encode input, x, to get mean and variance: encoder(x) = (mu, log(var))
        x = self.encoder(x) # shape (B, 64, 1, 1)
        x = x.view(x.size(0), -1) # shape (B, 64)
        
        mu = self.linear_mu(x) # shape (B, latent_dim)
        logvar = self.linear_logvar(x) # shape (B, latent_dim)
        
        # sample z from approximate posterior: z ~ q(z|x) = N(z; mu, var*I)
        # reparameterization trick: z = var * eps + mu, where eps ~ N(0, I)
        eps = torch.rand_like(mu)
        
        z = torch.exp(logvar) * eps + mu
        
        # generate data from z: y ~ p(x|z)
        z = self.linear_dec(z)
        x_hat = self.decoder(z) # shape (B, 1, 28, 28)
        
        return x_hat, mu, logvar
    
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, layers, bias=True):
        super(VariationalAutoencoder, self).__init__()
        
        self.layers = layers
        self.bias = bias
        
        encoder_layers = []
        decoder_layers = []
        
        if len(layers) == 2:
            encoder_layers.append(('layer1', nn.Linear(in_features=layers[0], out_features=layers[1], bias=bias)))
            decoder_layers.append(('layer1', nn.Linear(in_features=layers[1], out_features=layers[0], bias=bias)))
                    
        else:
            # encoder
            for i in range(len(layers) - 1):
                encoder_layers.append((f'layer{i+1}', nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=bias)))
                
                if i < len(layers) - 2:
                    encoder_layers.append((f'relu{i+1}', nn.ReLU()))
            
            # decoder
            for i in range(1, len(layers)):
                decoder_layers.append((f'layer{i}', nn.Linear(in_features=layers[-i], out_features=layers[-i-1], bias=bias)))
                
                if i < len(layers) - 1:
                    decoder_layers.append((f'relu{i}', nn.ReLU()))
                    
        decoder_layers.append(('sigmoid', nn.Sigmoid()))
            
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))
        
        self.linear_mu = nn.Linear(in_features=layers[-1], out_features=layers[-1])
        self.linear_logvar = nn.Linear(in_features=layers[-1], out_features=layers[-1])
        
        self.linear_dec = nn.Linear(in_features=layers[-1], out_features=layers[-1])
        
    def forward(self, x):
        """
        x.shape = (B, C, H, W)
        """
        input_shape = x.shape
        
        x = x.view(input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]) # shape (B, C*H*W)
        
        x = self.encoder(x) # shape (B, latent_dim)
        print(x.shape)
        
        mu = self.linear_mu(x) # shape (B, latent_dim)
        logvar = self.linear_logvar(x) # shape (B, latent_dim)
        
        e = torch.rand_like(mu) # shape (B, latent_dim)
        
        z = torch.exp(logvar) * e + mu # shape (B, latent_dim)
        z = self.linear_dec(z)
        
        x_hat = self.decoder(z) # shape (B, C*H*W)
        x_hat = x_hat.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) # shape (B, C, H, W)
        print(x_hat.shape)
        
        return x_hat, mu, logvar
    