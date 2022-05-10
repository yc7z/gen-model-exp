import torch
import torch.nn as nn
from collections import OrderedDict
    
 
class AutoencoderCNN(nn.Module):
    """A simple CNN Autoencoder for MNIST
    """
    
    def __init__(self):

        super(AutoencoderCNN, self).__init__()
        
        encoder_layers = []
        decoder_layers = []
        
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
        
    def forward(self, x):
        """
        x.shape = (B, 1, 28, 28)
        """
        
        x = self.encoder(x) # shape (B, 64, 1, 1)
        x = self.decoder(x) # shape (B, 1, 28, 28)
        
        return x    
    
    
class Autoencoder(nn.Module):
    """AutoEncoder Network with linear layers
    
    Args:
        layers: decreasing list of integers indicating the intermediate layer sizes of the encoder and decoder.
                If len(layers) == 2, returns a linear autoencoder. Otherwise returns a non-linear autoencoder (activations
                are applied between layers).
        bias (bool, optional): Include bias in each layer. Defaults to False.
    """
    
    def __init__(self, layers, bias=True):

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
            for i in range(len(layers) - 1):
                encoder_layers.append((f'layer{i+1}', nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=bias)))
                decoder_layers.append((f'layer{i+1}', nn.Linear(in_features=layers[-i-1], out_features=layers[-i-2], bias=bias)))
                
                if i < len(layers) - 2:
                    encoder_layers.append((f'relu{i+1}', nn.ReLU()))
                    decoder_layers.append((f'relu{i+1}', nn.ReLU()))
                    
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