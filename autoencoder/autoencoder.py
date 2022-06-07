import torch.nn as nn
from collections import OrderedDict
import numpy as np
    
 
class AutoencoderCNN(nn.Module):
    def __init__(self, latent_dim):
        super(AutoencoderCNN, self).__init__()  
        
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
        
    def forward(self, x):
        """
        x.shape = (B, 1, 28, 28)
        """
        
        x = self.encoder(x) # shape (B, latent_dim, 1, 1)
        x = self.decoder(x) # shape (B, 1, 28, 28)
        
        return x    
    
    
class Autoencoder(nn.Module):
    def __init__(self, layers=[28*28, 30], bias=True):
        super(Autoencoder, self).__init__()
        
        self.layers = layers
        self.bias = bias
        
        encoder_layers = []
        decoder_layers = []
        
        # linear autoencoder
        if len(layers) == 2:
            encoder_layers.append(('layer1', nn.Linear(in_features=layers[0], out_features=layers[1], bias=bias)))
            decoder_layers.append(('layer1', nn.Linear(in_features=layers[1], out_features=layers[0], bias=bias)))
        
        # nonlinear autoencoder
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
        
    def forward(self, x):
        """
        x.shape = (B, C, H, W)
        """
        input_shape = x.shape
        
        x = x.view(input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]) # shape (B, C*H*W)
        
        x = self.encoder(x) # shape (B, latent_dim)
        x = self.decoder(x) # shape (B, C*H*W)
        x = x.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) # shape (B, C, H, W)
        
        return x
    