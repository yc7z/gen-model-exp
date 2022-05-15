import torch
import torch.nn as nn

from collections import OrderedDict

class VariationalAutoencoderCNN(nn.Module):
    """A simple VAE for MNIST
    """
    def __init__(self, latent_dim):

        super(VariationalAutoencoderCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # shape (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # shape (B, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=1) # shape (B, 64, 1, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1), # (B, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # shape (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.linear_mu = nn.Linear(in_features=64, out_features=latent_dim)
        self.linear_logvar = nn.Linear(in_features=64, out_features=latent_dim)
        
        self.linear_dec = nn.Linear(in_features=latent_dim, out_features=64)
        
    def forward(self, x):
        """
        x.shape = (B, 1, 28, 28)
        """
        
        # encode input to get mean and variance
        x = self.encoder(x) # shape (B, 64, 1, 1)
        x = x.view(x.size(0), -1) # shape (B, 64)
        
        mu = self.linear_mu(x) # shape (B, latent_dim)
        logvar = self.linear_logvar(x) # shape (B, latent_dim)
        
        # sample latent vector using mean and variance
        e = torch.rand_like(mu)
        
        z = torch.exp(logvar) * e + mu
        
        y = self.decoder(z) # shape (B, 1, 28, 28)
        
        return y, mu, logvar
    
    
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
            for i in range(len(layers) - 1):
                encoder_layers.append((f'layer{i+1}', nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=bias)))
                decoder_layers.append((f'layer{i+1}', nn.Linear(in_features=layers[-i-1], out_features=layers[-i-2], bias=bias)))
                
                if i < len(layers) - 2:
                    encoder_layers.append((f'relu{i+1}', nn.ReLU()))
                    decoder_layers.append((f'relu{i+1}', nn.ReLU()))
                    
        decoder_layers.append(('sigmoid', nn.Sigmoid()))
            
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))
        
        self.linear_mu = nn.Linear(in_features=layers[-1], out_features=layers[-1])
        self.linear_logvar = nn.Linear(in_features=layers[-1], out_features=layers[-1])
        
    def forward(self, x):
        """
        x.shape = (B, C, H, W)
        """
        input_shape = x.shape
        
        x = x.view(input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]) # shape (B, C*H*W)
        
        x = self.encoder(x) # shape (B, latent_dim)
        
        mu = self.linear_mu(x) # shape (B, latent_dim)
        logvar = self.linear_logvar(x) # shape (B, latent_dim)
        
        e = torch.rand_like(mu) # shape (B, latent_dim)
        
        z = torch.exp(logvar) * e + mu # shape (B, latent_dim)
        
        y = self.decoder(z) # shape (B, C*H*W)
        y = y.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) # shape (B, C, H, W)
        
        return y, mu, logvar
    