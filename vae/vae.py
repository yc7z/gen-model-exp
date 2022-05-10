import torch
import torch.nn as nn

class VAE(nn.Module):
    """A simple VAE for MNIST
    """
    def __init__(self, latent_dim):

        super(VAE, self).__init__()
        
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